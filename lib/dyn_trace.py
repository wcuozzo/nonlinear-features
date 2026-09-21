"""Training-dynamics checkpoint recorder for the canonical sweep.

Records, at ~log-spaced steps during training:
  - W_eff [T, K, n, m] (fp16): per-feature effective encoder weights
    W_eff_i = enc(e_i) - enc(0). This is the quantity tms_learning_dynamics /
    frontier_dimensionality analyze (feature dimensionality D_i(t),
    crystallization timing, loss-drop <-> geometry-jump alignment).
  - losses [T, K] (fp32): the training-batch per-seed loss at that step.
  - full_states (optional, sparser cadence): full weight snapshots for
    downstream Hessian/LLC work. Enabled per (n, m, S) group only, since
    full weights for the whole fleet would be tens of GB of redundancy.

RNG-NEUTRALITY INVARIANT: recording consumes no random numbers (fixed-basis
forward passes and tensor copies only), so an instrumented run produces
bitwise-identical models to an uninstrumented one. Do not add fresh data
evals here — generate_sparse_data draws from the global RNG and would shift
every subsequent training batch.

One file per (config, phase, arm): {store}/dynamics/{phase}_n{n}_m{m}_l{l}_S{S}_{arm}.pt
"""

import numpy as np
import torch
from pathlib import Path


def log_spaced_steps(max_steps, n_ckpts=40):
    """~n_ckpts unique steps in [0, max_steps-1], log-spaced (dense early,
    where crystallization jumps happen)."""
    last = max(max_steps - 1, 1)
    pts = np.unique(np.geomspace(1, last, max(n_ckpts - 1, 1)).astype(int))
    return sorted({0, *pts.tolist(), last})


class DynamicsRecorder:
    def __init__(self, path, config, arm, seed_values, max_steps,
                 n_ckpts=40, full_weights=False, full_every=4, meta=None):
        self.path = Path(path)
        self.config = dict(config)          # n, m, l, S
        self.arm = arm
        self.seed_values = list(seed_values)
        self.steps_planned = log_spaced_steps(max_steps, n_ckpts)
        self._steps_set = set(self.steps_planned)
        # Full snapshots on every full_every-th planned checkpoint
        self.full_steps = (set(self.steps_planned[::full_every])
                           if full_weights else set())
        self.meta = dict(meta or {})
        self.steps, self.W_eff, self.losses = [], [], []
        self.full_states = []
        self._basis = None

    def wants(self, step):
        return step in self._steps_set

    @torch.no_grad()
    def record(self, step, model, mse_per_seed):
        """model: BatchedAutoencoder ([B,n]->[K,B,m] encode) or a single
        core.Autoencoder ([B,n]->[B,m] encode). mse_per_seed: [K] tensor
        (or scalar tensor for a single model) — the current batch loss."""
        if not self.wants(step):
            return
        n = self.config['n']
        dev = next(model.parameters()).device
        if self._basis is None or self._basis.device != dev:
            basis = torch.zeros(n + 1, n, device=dev)
            basis[:n] = torch.eye(n, device=dev)
            self._basis = basis
        z = model.encode(self._basis)
        if z.dim() == 2:                     # single model -> fake K=1
            z = z.unsqueeze(0)
        w_eff = (z[:, :n, :] - z[:, n:, :]).to(torch.float16).cpu()
        self.steps.append(int(step))
        self.W_eff.append(w_eff)
        self.losses.append(mse_per_seed.detach().float().reshape(-1).cpu())
        if step in self.full_steps:
            self.full_states.append(
                (int(step),
                 {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))

    def save(self, final_info=None):
        if final_info:
            self.meta['final_info'] = final_info
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(
            config=self.config,
            arm=self.arm,
            seed_values=self.seed_values,
            steps=self.steps,
            W_eff=torch.stack(self.W_eff) if self.W_eff else None,
            losses=torch.stack(self.losses) if self.losses else None,
            full_states=self.full_states,
            meta=self.meta,
        ), self.path)


def parse_full_groups(items):
    """Parse ['16,2,0.9', '64,8,0.9'] -> {(16, 2, 0.9), (64, 8, 0.9)}."""
    out = set()
    for it in items or []:
        n, m, S = it.split(',')
        out.add((int(n), int(m), float(S)))
    return out
