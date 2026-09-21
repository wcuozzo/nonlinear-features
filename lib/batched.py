"""
Batched-autoencoder LIBRARY: K seeds trained in one bmm-batched forward/backward.

Live pieces (imported by train_sweep / frontier_push / exploratory tools):
  - BatchedAutoencoder      — K packed seeds, [K, in, out] weights, untied default
  - measure_batched_linearity — hidden_state_linearity (affine x->z R^2) per seed
  - cosine_lr, set_device, generate_sparse_data (GPU-native)
  - validate()              — proves batched == standard layer-for-layer (--validate)

The original all-in-one sweep CLI that used to live here is DEPRECATED and frozen at
archive/run_sweep_gpu_full_sweep.py (salted-hash seeds and all). Canonical pipeline:
train_sweep.py (l=1..4 chain) -> frontier_push.py -> precise_recompile.py.
"""

import os
import sys
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    import matplotlib
    if 'IPython' not in sys.modules:
        # headless CLI sweeps want Agg; but don't hijack a notebook's inline backend
        # (an unconditional use('Agg') here silently killed figure capture on import)
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from core import Autoencoder, make_adamw

# Default device — overridden by --device flag or set_device()
device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

# Enable TF32 for A100s — ~3x faster matmuls with negligible precision loss
torch.set_float32_matmul_precision('high')

# The sweep compiles a fresh BatchedAutoencoder per (n, m, l, K) shape — far
# more than dynamo's default recompile limit (8), after which it SILENTLY runs
# eager. That both loses ~10-45% throughput and makes kernels (hence rounding)
# depend on how many shapes a worker saw first — a scheduling-dependent
# numerics channel. Raise the limits so every shape compiles, uniformly.
if torch.cuda.is_available():
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 512
    if hasattr(torch._dynamo.config, 'accumulated_cache_size_limit'):
        torch._dynamo.config.accumulated_cache_size_limit = 2048


def set_device(dev):
    """Set the device for this module and core.py."""
    global device
    import core
    device = torch.device(dev)
    core.device = device


# ════════════════════════════════════════════════════════════════════════
# Data generation (GPU-native: avoids CPU→GPU copy)
# ════════════════════════════════════════════════════════════════════════

def generate_sparse_data(n_samples, n_features, S):
    """Sparse data created directly on the active device."""
    mask = (torch.rand(n_samples, n_features, device=device) > S).float()
    values = torch.rand(n_samples, n_features, device=device)
    return mask * values


# ════════════════════════════════════════════════════════════════════════
# Batched Autoencoder
# ════════════════════════════════════════════════════════════════════════

class BatchedAutoencoder(nn.Module):
    """K autoencoders with identical (n, m, l), different initializations.

    Trains all K seeds in a single forward/backward pass using batched
    matrix multiplies (bmm). Weight tensors have shape [K, in, out].

    Architecture matches core.Autoencoder exactly:
      Encoder: (l-1) × [Linear(n,n) + ReLU] + Linear(n,m)  (bottleneck layer bias-free)
      Decoder: Linear(m,n) + (l-1) × [ReLU + Linear(n,n)] + ReLU
      Untied at every depth (default); pass tied=True for the paper's tied-l1.
    """

    def __init__(self, n, m, l, K, seeds=None, tied=False):
        super().__init__()
        self.n, self.m, self.l, self.K = n, m, l, K
        self.tied = tied

        if seeds is None:
            seeds = list(range(K))

        if self.tied:
            self.W = nn.Parameter(torch.empty(K, n, m))
            self.dec_bias = nn.Parameter(torch.zeros(K, 1, n))
        else:
            self.enc_W = nn.ParameterList()
            self.enc_bias = nn.ParameterList()
            for i in range(l - 1):
                self.enc_W.append(nn.Parameter(torch.empty(K, n, n)))
                self.enc_bias.append(nn.Parameter(torch.empty(K, 1, n)))
            self.enc_W.append(nn.Parameter(torch.empty(K, n, m)))  # bottleneck: bias-free

            self.dec_W = nn.ParameterList()
            self.dec_bias = nn.ParameterList()
            self.dec_W.append(nn.Parameter(torch.empty(K, m, n)))
            self.dec_bias.append(nn.Parameter(torch.empty(K, 1, n)))
            for i in range(l - 1):
                self.dec_W.append(nn.Parameter(torch.empty(K, n, n)))
                self.dec_bias.append(nn.Parameter(torch.empty(K, 1, n)))

        self._init_from_reference(seeds)

    def _init_from_reference(self, seeds):
        """Initialize each seed's weights from a standard Autoencoder.

        This guarantees identical initialization to the non-batched version,
        regardless of parameter storage layout differences.
        """
        with torch.no_grad():
            for k, seed in enumerate(seeds):
                torch.manual_seed(seed)
                ref = Autoencoder(self.n, self.m, self.l, tied_weights=self.tied)
                if self.tied:
                    # nn.Linear stores [out, in]; we store [in, out]
                    enc_lin = [m for m in ref.encoder if isinstance(m, nn.Linear)][-1]
                    self.W.data[k] = enc_lin.weight.data.T
                    self.dec_bias.data[k, 0] = ref.dec_biases[0].data
                else:
                    idx = 0
                    for layer in ref.encoder:
                        if isinstance(layer, nn.Linear):
                            self.enc_W[idx].data[k] = layer.weight.data.T
                            if layer.bias is not None:      # bottleneck layer is bias-free
                                self.enc_bias[idx].data[k, 0] = layer.bias.data
                            idx += 1
                    idx = 0
                    for layer in ref.decoder:
                        if isinstance(layer, nn.Linear):
                            self.dec_W[idx].data[k] = layer.weight.data.T
                            self.dec_bias[idx].data[k, 0] = layer.bias.data
                            idx += 1

    def encode(self, x):
        """x: [B, n] -> z: [K, B, m]"""
        h = x.unsqueeze(0).expand(self.K, -1, -1)  # [K, B, n]
        if self.tied:
            return torch.bmm(h, self.W)  # [K, B, m]
        for i in range(self.l - 1):
            h = torch.relu(torch.bmm(h, self.enc_W[i]) + self.enc_bias[i])
        return torch.bmm(h, self.enc_W[-1])  # bottleneck: bias-free

    def decode(self, z):
        """z: [K, B, m] -> x_hat: [K, B, n]"""
        if self.tied:
            return torch.relu(
                torch.bmm(z, self.W.transpose(1, 2)) + self.dec_bias
            )
        h = torch.bmm(z, self.dec_W[0]) + self.dec_bias[0]
        for i in range(1, self.l):
            h = torch.relu(h)
            h = torch.bmm(h, self.dec_W[i]) + self.dec_bias[i]
        return torch.relu(h)

    def forward(self, x):
        """x: [B, n] -> (x_hat: [K, B, n], z: [K, B, m])"""
        z = self.encode(x)
        return self.decode(z), z

    def extract_single(self, k):
        """Extract seed k as a standard core.Autoencoder (for metrics)."""
        model = Autoencoder(self.n, self.m, self.l, tied_weights=self.tied)
        model = model.to(next(self.parameters()).device)
        with torch.no_grad():
            if self.tied:
                enc_lin = [m for m in model.encoder if isinstance(m, nn.Linear)][-1]
                enc_lin.weight.data.copy_(self.W.data[k].T)
                model.dec_biases[0].data.copy_(self.dec_bias.data[k, 0])
            else:
                idx = 0
                for layer in model.encoder:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.copy_(self.enc_W[idx].data[k].T)
                        if layer.bias is not None:      # bottleneck layer is bias-free
                            layer.bias.data.copy_(self.enc_bias[idx].data[k, 0])
                        idx += 1
                idx = 0
                for layer in model.decoder:
                    if isinstance(layer, nn.Linear):
                        layer.weight.data.copy_(self.dec_W[idx].data[k].T)
                        layer.bias.data.copy_(self.dec_bias[idx].data[k, 0])
                        idx += 1
        return model


# ════════════════════════════════════════════════════════════════════════
# Batched linearity measurement
# ════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def measure_batched_linearity(model, n_samples=2000, S=0.9):
    """Hidden-state linearity (R^2 of best affine fit x -> z) for all K seeds at once.

    Returns dict with lists of length K:
        hidden_state_linearities, mse_fulls
    """
    model.eval()
    K = model.K

    x = generate_sparse_data(n_samples, model.n, S)         # [N, n]
    z = model.encode(x)                                      # [K, N, m]
    x_hat_full = model.decode(z)                              # [K, N, n]

    # Batched linear fit: z[k] ~ x_aug @ W_lin[k]
    ones = torch.ones(n_samples, 1, device=device)
    x_aug = torch.cat([x, ones], dim=1)                      # [N, n+1]
    x_aug_K = x_aug.unsqueeze(0).expand(K, -1, -1)           # [K, N, n+1]
    W_lin = torch.linalg.lstsq(x_aug_K, z).solution          # [K, n+1, m]
    z_lin = torch.bmm(x_aug_K, W_lin)                         # [K, N, m]

    # Hidden-state linearity: 1 - Var(residual) / Var(z)
    z_var = z.var(dim=1).sum(dim=1)                           # [K]
    res_var = (z - z_lin).var(dim=1).sum(dim=1)               # [K]
    linearity_scores = (1 - res_var / (z_var + 1e-10))        # [K]

    x_K = x.unsqueeze(0).expand(K, -1, -1)                   # [K, N, n]
    mse_full = ((x_hat_full - x_K) ** 2).mean(dim=(1, 2))    # [K]

    model.train()
    return {
        'hidden_state_linearities': linearity_scores.cpu().tolist(),
        'mse_fulls': mse_full.cpu().tolist(),
    }


# ════════════════════════════════════════════════════════════════════════
# LR schedule
# ════════════════════════════════════════════════════════════════════════

def cosine_lr(step, max_steps, lr_peak, warmup=1000, lr_min=1e-6):
    """Linear warmup then cosine decay."""
    warmup = min(warmup, max_steps // 10)
    if step < warmup:
        return lr_peak * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return lr_min + 0.5 * (lr_peak - lr_min) * (1 + math.cos(math.pi * progress))


# ════════════════════════════════════════════════════════════════════════
# Validation
# ════════════════════════════════════════════════════════════════════════

def validate():
    """Verify BatchedAutoencoder matches standard Autoencoder exactly."""
    print("=== Forward pass validation ===")

    cases = [(8, 3, 1), (8, 3, 2), (16, 4, 3), (32, 8, 4)]
    all_pass = True

    for n, m, l in cases:
        seed = 42
        torch.manual_seed(seed)
        std = Autoencoder(n, m, l).to(device)  # default untied (matches BatchedAutoencoder)

        batched = BatchedAutoencoder(n, m, l, K=1, seeds=[seed]).to(device)
        extracted = batched.extract_single(0)

        torch.manual_seed(0)
        x = generate_sparse_data(200, n, 0.9)

        std.eval(); batched.eval(); extracted.eval()
        with torch.no_grad():
            xh_std, z_std = std(x)
            xh_bat, z_bat = batched(x)
            xh_ext, z_ext = extracted(x)

        z_err = (z_std - z_bat[0]).abs().max().item()
        xh_err = (xh_std - xh_bat[0]).abs().max().item()
        ext_err = (xh_std - xh_ext).abs().max().item()
        ok = max(z_err, xh_err, ext_err) < 1e-5
        all_pass = all_pass and ok
        print(f"  n={n:3d} m={m:2d} l={l}: "
              f"z_err={z_err:.2e}  xh_err={xh_err:.2e}  "
              f"ext_err={ext_err:.2e}  [{'PASS' if ok else 'FAIL'}]")

    print("\n=== Training step validation (20 steps, K=1 vs standard) ===")
    n, m, l = 8, 3, 2
    seed = 42
    lr, wd = 1e-3, 1e-2

    torch.manual_seed(seed)
    std = Autoencoder(n, m, l, tied_weights=False).to(device)
    std_opt = make_adamw(std, lr=lr, weight_decay=wd)

    batched = BatchedAutoencoder(n, m, l, K=1, seeds=[seed]).to(device)
    bat_opt = make_adamw(batched, lr=lr, weight_decay=wd)

    for step in range(20):
        torch.manual_seed(1000 + step)
        x = generate_sparse_data(64, n, 0.9)

        # Standard step
        std_opt.zero_grad()
        xh_s, _ = std(x)
        loss_s = nn.functional.mse_loss(xh_s, x)
        loss_s.backward()
        std_opt.step()

        # Batched step (same data -- same seed, same device)
        bat_opt.zero_grad()
        xh_b, _ = batched(x)
        x_K = x.unsqueeze(0)
        loss_b = ((xh_b - x_K) ** 2).mean(dim=(1, 2)).sum()
        loss_b.backward()
        bat_opt.step()

    # Compare outputs after training
    ext = batched.extract_single(0)
    std.eval(); ext.eval()
    with torch.no_grad():
        torch.manual_seed(9999)
        x_test = generate_sparse_data(100, n, 0.9)
        xh_s, _ = std(x_test)
        xh_e, _ = ext(x_test)
        err = (xh_s - xh_e).abs().max().item()

    ok = err < 1e-4
    all_pass = all_pass and ok
    print(f"  After 20 steps: max output diff = {err:.2e}  "
          f"[{'PASS' if ok else 'WARN' if err < 1e-2 else 'FAIL'}]")

    print("\n=== Batched linearity measurement ===")
    n, m, l = 16, 4, 2
    seed = 42
    torch.manual_seed(seed)
    std = Autoencoder(n, m, l, tied_weights=False).to(device)
    # Quick train
    std_opt = make_adamw(std, lr=1e-3, weight_decay=1e-2)
    for _ in range(500):
        x = generate_sparse_data(256, n, 0.9)
        std_opt.zero_grad()
        xh, _ = std(x)
        nn.functional.mse_loss(xh, x).backward()
        std_opt.step()

    batched = BatchedAutoencoder(n, m, l, K=1, seeds=[seed]).to(device)
    # Copy trained weights into batched model
    with torch.no_grad():
        idx = 0
        for layer in std.encoder:
            if isinstance(layer, nn.Linear):
                batched.enc_W[idx].data[0] = layer.weight.data.T
                if layer.bias is not None:      # bottleneck layer is bias-free
                    batched.enc_bias[idx].data[0, 0] = layer.bias.data
                idx += 1
        idx = 0
        for layer in std.decoder:
            if isinstance(layer, nn.Linear):
                batched.dec_W[idx].data[0] = layer.weight.data.T
                batched.dec_bias[idx].data[0, 0] = layer.bias.data
                idx += 1

    torch.manual_seed(12345)
    from core import measure_encoding_linearity
    std_lin = measure_encoding_linearity(std, n_samples=2000, S=0.9)

    torch.manual_seed(12345)
    bat_lin = measure_batched_linearity(batched, n_samples=2000, S=0.9)

    gain_err = abs(std_lin['hidden_state_linearity'] - bat_lin['hidden_state_linearities'][0])
    mse_err = abs(std_lin['mse_full'] - bat_lin['mse_fulls'][0])
    ok = gain_err < 5e-3 and mse_err < 5e-3
    all_pass = all_pass and ok
    print(f"  gain err={gain_err:.2e}  mse err={mse_err:.2e}  "
          f"[{'PASS' if ok else 'FAIL'}]")

    print(f"\n{'All tests passed!' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


if __name__ == '__main__':
    import argparse as _ap
    p = _ap.ArgumentParser(description='Batched-autoencoder library. The full sweep CLI '
                           'is DEPRECATED — see archive/run_sweep_gpu_full_sweep.py; '
                           'canonical pipeline: run_pipeline.sh')
    p.add_argument('--validate', action='store_true', help='verify batched == standard')
    p.add_argument('--device', default=None)
    a = p.parse_args()
    if a.device:
        set_device(a.device)
    if a.validate:
        validate()
    else:
        p.print_help()
