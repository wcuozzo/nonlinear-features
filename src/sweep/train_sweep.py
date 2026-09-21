"""
Progressive-from-l=1 retraining for monotonicity violations.

KEY FIX vs sweep_progressive.py:
   - sweep_progressive.py loops `for l in range(2, max_l+1)`, starting from
     RANDOM init at l=2. It can't use l=1 because its warm-start function
     (`warm_start_batched`) requires both source and target to be non-tied
     (l >= 2). So in the linear regime — where the converged l=1 model is
     already near-optimal — that good baseline is thrown away, and l=2's
     random init can find a bad basin. Then l=3 inherits the bad basin,
     and l=4 inherits a worse one. Errors compound through the chain.

   - This script loops `for l in range(2, max_l+1)` but warm-starts l=2
     from the converged l=1 model via `embed_shallow_in_deep`, which
     handles the tied l=1 architecture correctly (the data is non-negative,
     so identity + ReLU is a true pass-through; the tied l=1's W goes into
     the deep encoder's last layer and W.T goes into the deep decoder's
     first layer).

   - At each stage l, K seeds are initialized with small perturbations of
     the warm-start. The best resulting model becomes the source for l+1.

Verification (MPS, 5 worst small-n violations, K=1, 15k steps each):
   mean improvement vs stored MSE: 57x. All 5 also beat the shallow floor.

Usage:
    python train_sweep.py --list-only
    python train_sweep.py --device mps --K 4 --max-n 32 --limit-groups 2
    python train_sweep.py --device cuda --K 20 --n-gpus 8
"""

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', 'lib'))
import core
from core import Autoencoder, det_seed, generate_sparse_data, make_adamw
from dyn_trace import parse_full_groups
from results_store import ResultsStore
from warm_start import (
    embed_shallow_in_deep,
    eval_mse,
    load_best_model,
)
from batched import BatchedAutoencoder, measure_batched_linearity

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


# ────────────────────────────────────────────────────────────────────────
# Violation discovery — groups by (n, m, S) with their max-failed-l
# ────────────────────────────────────────────────────────────────────────

def find_violations(store_dir: str = 'data_and_models/results_db'):
    """Return list of dicts: each violation as (n, m, l, S, gap)."""
    df = pd.read_csv(Path(store_dir) / 'compiled' / 'sweep_results.csv')
    df_idx = df.set_index(['n', 'm', 'l', 'S'])

    violations = {}
    for _, row in df.iterrows():
        n, m, l, S = int(row.n), int(row.m), int(row.l), row.S
        mse = row.mse_full
        for l2 in range(l + 1, 5):
            key = (n, m, l2, S)
            if key in df_idx.index:
                mse2 = float(df_idx.loc[key, 'mse_full'])
                if mse2 > mse * 1.001:
                    if key not in violations or mse < violations[key]['mse_target']:
                        violations[key] = dict(
                            mse_target=mse, mse_current=mse2, shallow_l=l,
                            gap=mse2 / mse,
                        )
    rows = [dict(n=k[0], m=k[1], l=k[2], S=k[3], **v) for k, v in violations.items()]
    return pd.DataFrame(rows).sort_values('gap', ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def group_violations_by_nms(violations_df):
    """Group violations by (n, m, S); return list of dicts with max_l."""
    if violations_df.empty:
        return []
    groups = {}
    for _, row in violations_df.iterrows():
        n, m, S = int(row.n), int(row.m), row.S
        key = (n, m, S)
        if key not in groups:
            groups[key] = dict(n=n, m=m, S=S, max_l=int(row.l),
                               max_gap=float(row.gap), violated_ls=[int(row.l)])
        else:
            groups[key]['max_l'] = max(groups[key]['max_l'], int(row.l))
            groups[key]['max_gap'] = max(groups[key]['max_gap'], float(row.gap))
            groups[key]['violated_ls'].append(int(row.l))
    # Sort by max gap (worst first)
    return sorted(groups.values(), key=lambda g: -g['max_gap'])


# ────────────────────────────────────────────────────────────────────────
# Training utilities
# ────────────────────────────────────────────────────────────────────────

def cosine_lr(step, max_steps, peak, warmup=1000, lr_min=1e-6):
    warmup = min(warmup, max_steps // 10)
    if step < warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return lr_min + 0.5 * (peak - lr_min) * (1 + math.cos(math.pi * progress))


def train_one_seed(model: Autoencoder, S: float, max_steps: int,
                   batch_size: int = 4096, lr_peak: float = 4e-3,
                   weight_decay: float = 1e-2,
                   floor_check_every: int = 1000) -> dict:
    """Train one autoencoder; keep best-ever weights as a floor."""
    init_mse = eval_mse(model, S, device=device)
    optimizer = make_adamw(model, lr=lr_peak, weight_decay=weight_decay)

    best_mse = init_mse
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    loss_curve = []

    for step in range(max_steps):
        lr = cosine_lr(step, max_steps, lr_peak)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = generate_sparse_data(batch_size, model.n, S)
        x_hat, _ = model(x)
        loss = ((x_hat - x) ** 2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            loss_curve.append((step, float(loss.item())))

        if step > 0 and step % floor_check_every == 0:
            cur_mse = eval_mse(model, S, device=device)
            if cur_mse < best_mse:
                best_mse = cur_mse
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Final floor enforcement
    final_mse = eval_mse(model, S, device=device)
    if final_mse > best_mse:
        model.load_state_dict(best_state)
        final_mse = best_mse

    # Full metrics (lstsq native on CUDA/CPU, falls back to CPU on MPS)
    model.eval()
    torch.manual_seed(99999)
    with torch.no_grad():
        x = generate_sparse_data(4000, model.n, S)
        z = model.encode(x)
        ones = torch.ones(x.shape[0], 1, device=device)
        x_aug = torch.cat([x, ones], dim=1)
        if device.type == 'mps':
            W_lin = torch.linalg.lstsq(x_aug.cpu(), z.cpu()).solution.to(device)
        else:
            W_lin = torch.linalg.lstsq(x_aug, z).solution
        z_lin = x_aug @ W_lin
        x_recon_full, _ = model(x)
        mse_full = ((x_recon_full - x) ** 2).mean().item()
        z_var = z.var(dim=0).sum().item()
        res_var = (z - z_lin).var(dim=0).sum().item()
        lin_score = 1 - res_var / (z_var + 1e-10)
    model.train()

    return dict(
        init_mse=init_mse,
        final_mse=final_mse,
        mse_full=mse_full,
        hidden_state_linearity=float(lin_score),
        loss_curve=loss_curve,
    )


# ────────────────────────────────────────────────────────────────────────
# Load a single Autoencoder's weights into a BatchedAutoencoder slot k
# (inverse of BatchedAutoencoder.extract_single)
# ────────────────────────────────────────────────────────────────────────

def load_seed_into_batched(batched: BatchedAutoencoder, k: int,
                            ae: Autoencoder):
    """Copy ae's weights into batched's k-th slot.

    nn.Linear stores weight as [out, in]; BatchedAutoencoder stores [K, in, out],
    so we transpose into slot k.
    """
    with torch.no_grad():
        if batched.tied:
            enc_lin = [m for m in ae.encoder if isinstance(m, nn.Linear)][-1]
            batched.W.data[k] = enc_lin.weight.data.T
            batched.dec_bias.data[k, 0] = ae.dec_biases[0].data
        else:
            idx = 0
            for layer in ae.encoder:
                if isinstance(layer, nn.Linear):
                    batched.enc_W[idx].data[k] = layer.weight.data.T
                    if layer.bias is not None:      # bottleneck layer is bias-free
                        batched.enc_bias[idx].data[k, 0] = layer.bias.data
                    idx += 1
            idx = 0
            for layer in ae.decoder:
                if isinstance(layer, nn.Linear):
                    batched.dec_W[idx].data[k] = layer.weight.data.T
                    batched.dec_bias[idx].data[k, 0] = layer.bias.data
                    idx += 1


def cosine_lr_inline(step, max_steps, peak, warmup=1000, lr_min=1e-6):
    warmup = min(warmup, max_steps // 10)
    if step < warmup:
        return peak * step / max(warmup, 1)
    progress = (step - warmup) / max(max_steps - warmup, 1)
    return lr_min + 0.5 * (peak - lr_min) * (1 + math.cos(math.pi * progress))


# ────────────────────────────────────────────────────────────────────────
# Shared pool training: train K seeds together via BatchedAutoencoder.
# Used by both train_stage (sweep) and frontier_push.push_one_config.
# ────────────────────────────────────────────────────────────────────────

def train_pool_batched(autoencoders, init_mses, seed_values,
                       n, m, l, S, max_steps, batch_size=4096,
                       lr_peak=4e-3, weight_decay=1e-2,
                       floor_check_every=1000,
                       grad_clip=None, ema_decay=None,
                       dyn_recorder=None):
    """Train a list of K Autoencoder inits in lockstep via BatchedAutoencoder.

    Returns dict with per-seed final metrics (mse_full, hidden_state_linearity),
    plus best_k / best_mse / best_model / losses_log / init_mses / seed_values.

    Floor enforcement: per-seed init MSE is the floor, plus the global-min snapshot
    is restored if final < it. Optional grad_clip + EMA (Polyak averaging).
    """
    K = len(autoencoders)
    batched = BatchedAutoencoder(n, m, l, K, seeds=seed_values).to(device)
    for k, ae in enumerate(autoencoders):
        load_seed_into_batched(batched, k, ae)

    optimizer = make_adamw(batched, lr=lr_peak, weight_decay=weight_decay)
    compiled_model = (torch.compile(batched, mode='reduce-overhead')
                      if device.type == 'cuda' else batched)

    best_mse_per_seed = list(init_mses)
    best_overall_mse = float(min(init_mses))
    best_state_dict = {k: v.clone() for k, v in batched.state_dict().items()}
    losses_log = []

    ema_state = None
    if ema_decay is not None:
        ema_state = {k: v.clone().detach() for k, v in batched.state_dict().items()}

    for step in range(max_steps):
        lr = cosine_lr_inline(step, max_steps, lr_peak)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        x = generate_sparse_data(batch_size, n, S)
        x_hat, _ = compiled_model(x)
        x_K = x.unsqueeze(0).expand(K, -1, -1)
        mse_per_seed = ((x_hat - x_K) ** 2).mean(dim=(1, 2))
        loss = mse_per_seed.sum()

        # Pre-update snapshot (RNG-neutral; uses the uncompiled module)
        if dyn_recorder is not None and dyn_recorder.wants(step):
            dyn_recorder.record(step, batched, mse_per_seed)

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(batched.parameters(), grad_clip)
        optimizer.step()

        if ema_state is not None:
            with torch.no_grad():
                for kk, vv in batched.state_dict().items():
                    ema_state[kk].mul_(ema_decay).add_(vv.detach(), alpha=1 - ema_decay)

        if step % 400 == 0:
            losses_log.append((step, mse_per_seed.detach().cpu().tolist()))

        if step > 0 and step % floor_check_every == 0:
            torch.manual_seed(99999)
            lin = measure_batched_linearity(batched, n_samples=2000, S=S)
            for k in range(K):
                if lin['mse_fulls'][k] < best_mse_per_seed[k]:
                    best_mse_per_seed[k] = lin['mse_fulls'][k]
            cur_best = min(lin['mse_fulls'])
            if cur_best < best_overall_mse:
                best_overall_mse = cur_best
                best_state_dict = {k: v.clone() for k, v in batched.state_dict().items()}

    # Final floor + optional EMA check
    torch.manual_seed(99999)
    final_lin = measure_batched_linearity(batched, n_samples=2000, S=S)
    if ema_state is not None:
        cur_state = {k: v.clone() for k, v in batched.state_dict().items()}
        batched.load_state_dict(ema_state)
        torch.manual_seed(99999)
        ema_lin = measure_batched_linearity(batched, n_samples=2000, S=S)
        if min(ema_lin['mse_fulls']) < min(final_lin['mse_fulls']):
            final_lin = ema_lin  # keep EMA loaded
        else:
            batched.load_state_dict(cur_state)
    if best_state_dict is not None:
        if min(final_lin['mse_fulls']) > best_overall_mse:
            batched.load_state_dict(best_state_dict)
            torch.manual_seed(99999)
            final_lin = measure_batched_linearity(batched, n_samples=2000, S=S)

    final_mses = final_lin['mse_fulls']
    best_k = int(np.argmin(final_mses))
    best_mse = float(final_mses[best_k])
    best_model = batched.extract_single(best_k).to(device)

    if dyn_recorder is not None:
        dyn_recorder.save(final_info=dict(
            final_mses=[float(v) for v in final_mses],
            init_mses=[float(v) for v in init_mses],
            best_k=best_k))

    result = dict(
        final_mses=final_mses,
        final_hidden_linearities=final_lin['hidden_state_linearities'],
        best_k=best_k,
        best_mse=best_mse,
        best_model=best_model,
        losses_log=losses_log,
        init_mses=init_mses,
        seed_values=seed_values,
    )

    del batched, compiled_model, optimizer, best_state_dict
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return result


# ────────────────────────────────────────────────────────────────────────
# Near-warm-start seed builder
# ────────────────────────────────────────────────────────────────────────

def build_near_warm_start_pool(source_model: Autoencoder, target_l: int,
                                n: int, m: int, S: float, K_nws: int,
                                noise_lo: float = 1e-7, noise_hi: float = 1e-2,
                                base_seed_tag=None):
    """Build K_nws Autoencoders, each a zero-noise embed of source_model + tiny
    log-uniform perturbation in [noise_lo, noise_hi]. Aimed at near-basin
    exploration without leaving the basin (paired with reduced LR in training).

    Returns (autoencoders, init_mses, seed_values, descriptions, noises).
    """
    if base_seed_tag is None:
        base_seed_tag = (n, m, target_l, S, 'nws')
    base_seed = det_seed(*base_seed_tag, mod=1000000)

    if K_nws == 1:
        noises = [math.sqrt(noise_lo * noise_hi)]
    else:
        noises = np.geomspace(noise_lo, noise_hi, K_nws).tolist()

    autoencoders, init_mses, seed_values, descriptions = [], [], [], []
    for k in range(K_nws):
        noise = float(noises[k])
        seed = base_seed + k
        ae = embed_shallow_in_deep(source_model, target_l,
                                   noise=noise, seed=seed, device=device)
        autoencoders.append(ae)
        init_mses.append(eval_mse(ae, S, device=device))
        seed_values.append(seed)
        descriptions.append(
            f'near_warm_start(l={source_model.l},noise={noise:.1e})')
    return autoencoders, init_mses, seed_values, descriptions, noises


# ────────────────────────────────────────────────────────────────────────
# Batched per-stage training: K seeds in one bmm forward, with floor enforcement
# ────────────────────────────────────────────────────────────────────────

def train_stage(source_model: Autoencoder, target_l: int, n: int, m: int,
                S: float, K: int, max_steps: int, batch_size: int,
                store: ResultsStore, run_id: str, group_key: str,
                lr_peak: float = 4e-3, weight_decay: float = 1e-2,
                floor_check_every: int = 1000,
                near_warm_start_K: int = None,
                near_warm_start_lr_mult: float = 0.3,
                explore_K: int = 2,
                master_seed: int = 42,
                dyn_dir=None, dyn_full: bool = False,
                verbose: bool = True) -> Autoencoder:
    """Train K seeds simultaneously via BatchedAutoencoder.

    Two arms:
      - LARGE-NOISE (K seeds): embed_shallow_in_deep with noise in {0.0..0.01},
        trained at lr_peak. Broad basin exploration.
      - NEAR-WARM-START (near_warm_start_K seeds): zero-noise embed +
        log-uniform tiny perturbation 1e-7..1e-2, trained at
        lr_peak * near_warm_start_lr_mult. Refines without leaving the basin.

    Defaults: near_warm_start_K = max(2, K // 2). Set near_warm_start_K=0 to
    disable the second arm (original single-arm behavior).
    """
    if near_warm_start_K is None:
        near_warm_start_K = max(2, K // 2)

    # ── ARM 1: large-noise warm-start ────────────────────────────────
    noise_schedule = [0.0, 0.0, 0.001, 0.001, 0.003, 0.003, 0.01, 0.01]
    ae_large, init_large, seed_large, desc_large, noise_large = [], [], [], [], []
    for k in range(K):
        noise = noise_schedule[k % len(noise_schedule)]
        seed_value = det_seed(n, m, target_l, S, k, group_key, master_seed,
                              mod=100000)
        ae = embed_shallow_in_deep(source_model, target_l,
                                   noise=noise, seed=seed_value,
                                   device=device)
        ae_large.append(ae)
        init_large.append(eval_mse(ae, S, device=device))
        seed_large.append(seed_value)
        desc_large.append(f'large_noise(l={source_model.l},noise={noise})')
        noise_large.append(noise)

    def _recorder(arm_label, seed_vals, arm_lr):
        if dyn_dir is None:
            return None
        from dyn_trace import DynamicsRecorder
        return DynamicsRecorder(
            Path(dyn_dir) / f'chain_n{n}_m{m}_l{target_l}_S{S}_{arm_label}.pt',
            config=dict(n=n, m=m, l=target_l, S=S),
            arm=arm_label, seed_values=seed_vals, max_steps=max_steps,
            full_weights=dyn_full,
            meta=dict(run_id=run_id, lr_peak=arm_lr,
                      source_l=source_model.l, phase='chain'))

    res_large = train_pool_batched(ae_large, init_large, seed_large,
                                   n, m, target_l, S, max_steps,
                                   batch_size=batch_size,
                                   lr_peak=lr_peak,
                                   weight_decay=weight_decay,
                                   floor_check_every=floor_check_every,
                                   dyn_recorder=_recorder('large', seed_large,
                                                          lr_peak))
    del ae_large

    # ── ARM 2: near-warm-start (optional) ────────────────────────────
    res_nws = None
    if near_warm_start_K > 0:
        ae_nws, init_nws, seed_nws, desc_nws, noise_nws = build_near_warm_start_pool(
            source_model, target_l, n, m, S, near_warm_start_K,
            base_seed_tag=(n, m, target_l, S, group_key, 'nws', master_seed))
        res_nws = train_pool_batched(ae_nws, init_nws, seed_nws,
                                     n, m, target_l, S, max_steps,
                                     batch_size=batch_size,
                                     lr_peak=lr_peak * near_warm_start_lr_mult,
                                     weight_decay=weight_decay,
                                     floor_check_every=floor_check_every,
                                     dyn_recorder=_recorder(
                                         'nws', seed_nws,
                                         lr_peak * near_warm_start_lr_mult))
        del ae_nws

    # ── ARM 3: EXPLORE — fresh random inits (explore-exploit: warm starts
    # exploit known basins; a few cold seeds per stage keep checking whether a
    # basin family unreachable from the chain beats them) ─────────────────
    res_rand = None
    if explore_K > 0:
        ae_rand, init_rand, seed_rand, desc_rand, noise_rand = [], [], [], [], []
        for k in range(explore_K):
            seed_value = det_seed(n, m, target_l, S, k, group_key, 'explore',
                                  master_seed, mod=100000)
            torch.manual_seed(seed_value)
            ae = Autoencoder(n, m, target_l, tied_weights=False).to(device)
            ae_rand.append(ae)
            init_rand.append(eval_mse(ae, S, device=device))
            seed_rand.append(seed_value)
            desc_rand.append('random_init')
            noise_rand.append(0.0)
        res_rand = train_pool_batched(ae_rand, init_rand, seed_rand,
                                      n, m, target_l, S, max_steps,
                                      batch_size=batch_size, lr_peak=lr_peak,
                                      weight_decay=weight_decay,
                                      floor_check_every=floor_check_every,
                                      dyn_recorder=_recorder('explore', seed_rand,
                                                             lr_peak))
        del ae_rand

    # ── Pick global best across arms ─────────────────────────────────
    arms = [
        dict(res=res_large, desc=desc_large, noises=noise_large,
             init_mses=init_large, label='large_noise',
             lr_peak=lr_peak, source_l=source_model.l),
    ]
    if res_nws is not None:
        arms.append(dict(res=res_nws, desc=desc_nws, noises=noise_nws,
                         init_mses=init_nws, label='near_warm_start',
                         lr_peak=lr_peak * near_warm_start_lr_mult,
                         source_l=source_model.l))
    if res_rand is not None:
        arms.append(dict(res=res_rand, desc=desc_rand, noises=noise_rand,
                         init_mses=init_rand, label='explore_random',
                         lr_peak=lr_peak, source_l=0))

    best_arm_idx = min(range(len(arms)), key=lambda i: arms[i]['res']['best_mse'])
    best_arm = arms[best_arm_idx]
    best_model = best_arm['res']['best_model']
    best_mse = best_arm['res']['best_mse']

    # ── Save seed_results from BOTH arms to store ────────────────────
    seed_results = []
    for arm in arms:
        res = arm['res']
        for k in range(len(res['final_mses'])):
            per_seed_loss_curve = [(s, vs[k]) for s, vs in res['losses_log'][::3]]
            seed_results.append(dict(
                seed_value=res['seed_values'][k],
                mse_full=float(res['final_mses'][k]),
                hidden_state_linearity=float(res['final_hidden_linearities'][k]),
                converged=True,
                steps_used=max_steps,
                loss_curve=per_seed_loss_curve,
                init_mse=float(arm['init_mses'][k]),
                warm_start_noise=float(arm['noises'][k]),
                warm_start_source_l=arm['source_l'],
                warm_start_arm=arm['label'],
                warm_start_source=arm['desc'][k],
                arm_lr_peak=arm['lr_peak'],
            ))

    training_meta = dict(
        method='progressive_from_l1_batched',
        target_l=target_l,
        source_l=source_model.l,
        max_steps=max_steps,
        batch_size=batch_size,
        K=K,
        near_warm_start_K=near_warm_start_K,
        near_warm_start_lr_mult=near_warm_start_lr_mult,
        lr_peak=lr_peak,
        group_key=group_key,
    )

    store.add_seeds(n, m, target_l, S, seed_results,
                    run_id=run_id,
                    model_state_dict=best_model.cpu().state_dict(),
                    training_meta=training_meta)
    best_model = best_model.to(device)

    if verbose:
        arm_str = ', '.join(
            f'{a["label"]}_K={len(a["res"]["final_mses"])}_best={a["res"]["best_mse"]:.5f}'
            for a in arms)
        print(f'    [l={target_l}] best_mse={best_mse:.5f} '
              f'(winning arm={best_arm["label"]}); '
              f'source_l={source_model.l}; {arm_str}')

    return best_model, best_mse


# ────────────────────────────────────────────────────────────────────────
# Process one (n, m, S) group: progressive chain from l=1
# ────────────────────────────────────────────────────────────────────────

def get_steps_for_n(n: int, base: int = 12_000) -> int:
    """Larger n gets more steps. Calibrated for MPS-friendly small n;
    GPU runs would scale up."""
    return int(base * math.sqrt(max(1, n / 16)))


def train_l1_stage(n: int, m: int, S: float, K: int, max_steps: int,
                   batch_size: int, store: ResultsStore, run_id: str,
                   group_key: str, lr_peak: float = 4e-3,
                   weight_decay: float = 1e-2, master_seed: int = 42,
                   dyn_dir=None, dyn_full: bool = False,
                   verbose: bool = True):
    """Stage l=1 of the progressive chain: K RANDOM-init l=1 seeds in lockstep.

    Absorbed from the old bootstrap_l1.py — l=1 is just the chain's first stage,
    not a separate tool. No shallower source exists, so the pool is pure random
    inits (det_seed-derived), trained with the SAME batched machinery and the
    SAME per-n step budget as every other stage.
    """
    ae_pool, init_mses, seed_values = [], [], []
    for k in range(K):
        seed_value = det_seed(n, m, 1, S, k, group_key, 'l1_random',
                              master_seed, mod=100000)
        torch.manual_seed(seed_value)
        ae = Autoencoder(n, m, 1, tied_weights=False).to(device)
        ae_pool.append(ae)
        init_mses.append(eval_mse(ae, S, device=device))
        seed_values.append(seed_value)

    recorder = None
    if dyn_dir is not None:
        from dyn_trace import DynamicsRecorder
        recorder = DynamicsRecorder(
            Path(dyn_dir) / f'chain_n{n}_m{m}_l1_S{S}_random.pt',
            config=dict(n=n, m=m, l=1, S=S), arm='l1_random',
            seed_values=seed_values, max_steps=max_steps,
            full_weights=dyn_full,
            meta=dict(run_id=run_id, lr_peak=lr_peak, source_l=0,
                      phase='chain'))

    res = train_pool_batched(ae_pool, init_mses, seed_values,
                             n, m, 1, S, max_steps,
                             batch_size=batch_size, lr_peak=lr_peak,
                             weight_decay=weight_decay,
                             dyn_recorder=recorder)
    del ae_pool

    seed_results = []
    for k in range(len(res['final_mses'])):
        per_seed_loss_curve = [(st, vs[k]) for st, vs in res['losses_log'][::3]]
        seed_results.append(dict(
            seed_value=res['seed_values'][k],
            mse_full=float(res['final_mses'][k]),
            hidden_state_linearity=float(res['final_hidden_linearities'][k]),
            converged=True,
            steps_used=max_steps,
            loss_curve=per_seed_loss_curve,
            init_mse=float(init_mses[k]),
            warm_start_arm='l1_random',
            warm_start_source='random_init',
        ))
    store.add_seeds(n, m, 1, S, seed_results, run_id=run_id,
                    model_state_dict=res['best_model'].cpu().state_dict(),
                    training_meta=dict(method='l1_random_batched', K=K,
                                       max_steps=max_steps,
                                       batch_size=batch_size,
                                       lr_peak=lr_peak, group_key=group_key))
    best_model = res['best_model'].to(device)
    if verbose:
        print(f'    [l=1] best_mse={res["best_mse"]:.5f} (random init, K={K})')
    return best_model, res['best_mse']


def find_chain_start(n: int, m: int, S: float, violated_ls,
                     models_dir: str = 'data_and_models/results_db/models'):
    """Find the highest l' such that (n, m, l', S) is already monotonic-clean
    and there's a model on disk for it. Returns (start_model, start_l, start_mse).

    "Monotonic-clean" means l' is not itself violated (not in `violated_ls`)
    AND its MSE is <= MSE of all lower l' (since the chain we start needs to
    inherit a real floor).
    """
    violated_set = set(violated_ls)

    # Walk down from max_l - 1 to 1, find highest non-violated with monotonic floor
    mses = {}
    best_start_l = None
    best_start_model = None
    best_start_mse = None

    for l_candidate in range(max(violated_ls) - 1, 0, -1):
        if l_candidate in violated_set:
            continue
        m_l = load_best_model(n, m, l_candidate, S, models_dir, device=device)
        if m_l is None:
            continue
        mse_l = eval_mse(m_l, S, device=device)
        mses[l_candidate] = mse_l

        # Need to verify this l is monotonic-clean against all shallower ls.
        # Specifically: must be >= the min MSE achievable by a converged shallow.
        # We'll check: load each l' < l_candidate, ensure m_l's MSE <= theirs.
        is_clean = True
        for l_shallower in range(1, l_candidate):
            m_s = load_best_model(n, m, l_shallower, S, models_dir, device=device)
            if m_s is None:
                continue
            mse_s = eval_mse(m_s, S, device=device)
            if mse_l > mse_s * 1.001:
                is_clean = False
                break

        if is_clean:
            best_start_l = l_candidate
            best_start_model = m_l
            best_start_mse = mse_l
            break

    if best_start_l is None:
        # Fall back to l=1
        m1 = load_best_model(n, m, 1, S, models_dir, device=device)
        if m1 is None:
            return None, None, None
        return m1, 1, eval_mse(m1, S, device=device)
    return best_start_model, best_start_l, best_start_mse


def fix_group(n: int, m: int, S: float, max_l: int, K: int,
              violated_ls,
              store: ResultsStore, batch_size: int = 4096,
              models_dir: str = 'data_and_models/results_db/models',
              run_id: str = None, verbose: bool = True,
              near_warm_start_K: int = None,
              near_warm_start_lr_mult: float = 0.3,
              explore_K: int = 2,
              master_seed: int = 42,
              steps_base: int = 12_000,
              dyn_dir=None, dyn_full_groups=frozenset()) -> dict:
    """Run the FULL progressive chain l=1 → l=2 → ... → max_l for one
    (n, m, S) group, regardless of which intermediate l's already look "clean".

    Rationale: even a stored l=2 with mse(l=2) ≤ mse(l=1) might be locally
    suboptimal — it was found by random init, not by warm-starting from l=1.
    Warm-starting l=2 from l=1's identity floor often finds a strictly better
    l=2, which then gives a better seed for l=3, etc. The additive store keeps
    whichever (old random or new warm-start) was best, so this can only help.
    """
    if run_id is None:
        run_id = f'progv2_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}'
    group_key = f'n{n}_m{m}_S{S}'

    # Per-group RNG seeding: the training-batch stream for this group is a
    # function of (master_seed, n, m, S) only — independent of which worker
    # picks it up or in what order groups run.
    torch.manual_seed(det_seed('group_rng', n, m, S, master_seed))
    np.random.seed(det_seed('group_rng_np', n, m, S, master_seed))

    max_steps = get_steps_for_n(n, base=steps_base)

    source_model = load_best_model(n, m, 1, S, models_dir, device=device)
    if source_model is None:
        # Stage l=1: no model on disk -> train it here (chain's first stage).
        if verbose:
            print(f'  ── Stage l=1 (random init; nothing shallower exists) ──')
        source_model, source_mse = train_l1_stage(
            n, m, S, K=K, max_steps=max_steps, batch_size=batch_size,
            store=store, run_id=run_id, group_key=group_key,
            master_seed=master_seed, dyn_dir=dyn_dir,
            dyn_full=(n, m, S) in dyn_full_groups, verbose=verbose)
    else:
        source_mse = eval_mse(source_model, S, device=device)

    if verbose:
        print(f'  Chain starts at l=1 (MSE={source_mse:.5f}); '
              f'will train l=2..{max_l}')

    stage_results = {1: source_mse}

    for target_l in range(2, max_l + 1):
        if verbose:
            print(f'  ── Stage l={target_l} (warm-start from l={source_model.l}) ──')
        new_model, new_mse = train_stage(
            source_model, target_l, n, m, S, K=K, max_steps=max_steps,
            batch_size=batch_size, store=store, run_id=run_id,
            group_key=group_key, verbose=verbose,
            near_warm_start_K=near_warm_start_K,
            near_warm_start_lr_mult=near_warm_start_lr_mult,
            explore_K=explore_K,
            master_seed=master_seed,
            dyn_dir=dyn_dir,
            dyn_full=(n, m, S) in dyn_full_groups)
        stage_results[target_l] = new_mse

        # CRITICAL: for next stage, warm-start from the actual on-disk best at
        # this l (which may be OLD random-init or NEW from this run). The store
        # only updates the .pt file when new best <= overall best, so reading
        # back gives us the true best.
        disk_best = load_best_model(n, m, target_l, S, models_dir, device=device)
        if disk_best is not None:
            disk_mse = eval_mse(disk_best, S, device=device)
            if disk_mse < new_mse:
                if verbose:
                    print(f'      [chain] using DISK best for next stage: '
                          f'{disk_mse:.5f} < my new {new_mse:.5f}')
                source_model = disk_best
                stage_results[target_l] = disk_mse
            else:
                source_model = new_model
        else:
            source_model = new_model

    return dict(n=n, m=m, S=S, stage_results=stage_results)


# ────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────

def _gpu_worker(gpu_id, task_queue, result_queue, K, batch_size, store_dir,
                near_warm_start_K=None, near_warm_start_lr_mult=0.3,
                explore_K=2,
                master_seed=42, run_id=None, steps_base=12_000,
                record_dynamics=False, dyn_full_groups=frozenset()):
    """One worker per GPU: pulls (n, m, S, max_l, violated_ls) groups and fixes them."""
    global device
    import core as _core
    import batched as _sweep
    device = torch.device(f'cuda:{gpu_id}')
    _core.device = device
    _sweep.device = device       # measure_batched_linearity uses this
    torch.cuda.set_device(gpu_id)
    # Per-worker deterministic seed: same master_seed -> same per-worker batches.
    torch.manual_seed(master_seed + gpu_id)
    np.random.seed(master_seed + gpu_id)

    store = ResultsStore(store_dir)
    while True:
        item = task_queue.get()
        if item is None:
            break
        idx, n, m, S, max_l, violated_ls = item
        try:
            t0 = time.time()
            summary = fix_group(n, m, S, max_l, K=K,
                                violated_ls=violated_ls, store=store,
                                batch_size=batch_size,
                                models_dir=os.path.join(store_dir, 'models'),
                                verbose=True,
                                near_warm_start_K=near_warm_start_K,
                                near_warm_start_lr_mult=near_warm_start_lr_mult,
                                explore_K=explore_K,
                                run_id=run_id,
                                master_seed=master_seed,
                                steps_base=steps_base,
                                dyn_dir=(os.path.join(store_dir, 'dynamics')
                                         if record_dynamics else None),
                                dyn_full_groups=dyn_full_groups)
            summary['elapsed_sec'] = time.time() - t0
            summary['gpu_id'] = gpu_id
            result_queue.put((idx, summary))
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            result_queue.put((idx, dict(n=n, m=m, S=S, error=str(e),
                                        traceback=tb, gpu_id=gpu_id)))


def main():
    parser = argparse.ArgumentParser(
        description='Progressive-from-l=1 violation fix')
    parser.add_argument('--store-dir', default='data_and_models/results_db')
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--K', type=int, default=6,
                        help='seeds per stage')
    parser.add_argument('--device', default=None,
                        help='Force device: cuda, mps, cpu, or cuda:N. '
                             'Ignored when --n-gpus > 1.')
    parser.add_argument('--n-gpus', type=int, default=1,
                        help='Number of GPUs for parallel groups')
    parser.add_argument('--max-n', type=int, default=None,
                        help='Only fix groups with n <= max-n')
    parser.add_argument('--limit-groups', type=int, default=None,
                        help='Only fix this many groups (worst first)')
    parser.add_argument('--explore-K', type=int, default=2,
                        help='Fresh RANDOM inits trained alongside the warm-start arms '
                             'at every chain stage (explore-exploit). 0 disables.')
    parser.add_argument('--near-warm-start-K', type=int, default=None,
                        help='Additional near-warm-start seeds per stage '
                             '(log-uniform 1e-7 to 1e-2 perturbation, reduced LR). '
                             'Default: max(2, K // 2). Set to 0 to disable.')
    parser.add_argument('--near-warm-start-lr-mult', type=float, default=0.3,
                        help='LR multiplier for the near-warm-start arm '
                             '(applied to lr_peak). Default 0.3.')
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--master-seed', type=int, default=42,
                        help='Master seed for batch sampling. Same seed -> same models.')
    parser.add_argument('--run-id', default=None,
                        help='Run id tag stored with each seed record.')
    parser.add_argument('--steps-base', type=int, default=12_000,
                        help='Base steps at n=16; scaled by sqrt(n/16). '
                             'Lower for smoke tests only.')
    parser.add_argument('--record-dynamics', action='store_true',
                        help='Record W_eff/loss training traces at log-spaced '
                             'steps into {store}/dynamics/ (RNG-neutral).')
    parser.add_argument('--dyn-full-groups', nargs='*', default=[],
                        help='Space-separated n,m,S groups that ALSO get full '
                             'weight snapshots (e.g. 16,2,0.9 64,8,0.9).')
    parser.add_argument('--resume', action='store_true',
                        help='Skip groups whose max-l stage already has chain '
                             'seed records tagged with --run-id (crash recovery; '
                             'sound because groups are deterministic and '
                             'independent). Requires --run-id.')

    parser.add_argument('--all-groups', action='store_true',
                        help='Iterate over EVERY (n, m, S) in the canonical sweep '
                             'rather than only violation groups. Use this for a fresh '
                             'single-sweep run.')
    parser.add_argument('--sweep-ns', default='16,32,64,128',
                        help='Comma-separated n values (with --all-groups). '
                             'Default: 16,32,64,128')
    parser.add_argument('--sweep-ms', default='2,4,8,16,32,64',
                        help='Comma-separated m values (with --all-groups). '
                             'Default: 2,4,8,16,32,64')
    parser.add_argument('--sweep-Ss', default='0.85,0.9,0.95',
                        help='Comma-separated S values (with --all-groups). '
                             'Default: 0.85,0.9,0.95')
    args = parser.parse_args()

    global device
    if args.device and args.n_gpus <= 1:
        device = torch.device(args.device)
        core.device = device
        import batched as _sweep
        _sweep.device = device
    print(f'Device: {device}, n_gpus: {args.n_gpus}')

    if args.all_groups:
        ns = [int(x) for x in args.sweep_ns.split(',')]
        ms = [int(x) for x in args.sweep_ms.split(',')]
        Ss = [float(x) for x in args.sweep_Ss.split(',')]
        # max_l=4 fixed; violated_ls=[2,3,4] forces the chain to retrain every l>=2
        # since all of l=2,3,4 are "treated as violations" → fix_group runs them all
        groups = []
        for n in ns:
            for m in ms:
                if m >= n:
                    continue
                for S in Ss:
                    groups.append(dict(n=n, m=m, S=S, max_l=4,
                                       max_gap=0.0, violated_ls=[2, 3, 4]))
        # Most expensive first: minimizes multi-GPU tail latency. Safe to
        # reorder — per-group results are scheduling-independent.
        groups.sort(key=lambda g: (-g['n'], -g['m']))
    else:
        violations = find_violations(args.store_dir)
        if violations.empty:
            print('No violations to fix!')
            return
        groups = group_violations_by_nms(violations)

    # Filter
    if args.max_n is not None:
        groups = [g for g in groups if g['n'] <= args.max_n]
    if args.limit_groups is not None:
        groups = groups[:args.limit_groups]

    if args.resume:
        if not args.run_id:
            raise SystemExit('--resume requires --run-id')
        import json

        def _chain_done(g):
            path = (Path(args.store_dir) / 'seeds'
                    / f"n{g['n']}_m{g['m']}_l{g['max_l']}_S{g['S']}.json")
            if not path.exists():
                return False
            for s in json.load(open(path)).get('seeds', []):
                if (s.get('run_id') == args.run_id and
                        s.get('warm_start_arm') in ('large_noise',
                                                    'near_warm_start')):
                    return True
            return False

        before = len(groups)
        groups = [g for g in groups if not _chain_done(g)]
        print(f'Resume: skipping {before - len(groups)} completed groups, '
              f'{len(groups)} remain')

    print(f'Groups to fix: {len(groups)}')
    for g in groups:
        print(f'  n={g["n"]:3d} m={g["m"]:2d} S={g["S"]} '
              f'max_l={g["max_l"]} max_gap={g["max_gap"]:.1f}x '
              f'violated_ls={sorted(set(g["violated_ls"]))}')

    if args.list_only:
        return

    start = time.time()

    if args.n_gpus > 1:
        # ─── Multi-GPU: spawn workers ───────────────────────────────
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()

        for idx, g in enumerate(groups):
            task_queue.put((idx, g['n'], g['m'], g['S'], g['max_l'],
                           sorted(set(g['violated_ls']))))
        for _ in range(args.n_gpus):
            task_queue.put(None)

        workers = []
        for gpu_id in range(args.n_gpus):
            p = ctx.Process(
                target=_gpu_worker,
                args=(gpu_id, task_queue, result_queue,
                      args.K, args.batch_size, args.store_dir,
                      args.near_warm_start_K, args.near_warm_start_lr_mult,
                      args.explore_K,
                      args.master_seed, args.run_id, args.steps_base,
                      args.record_dynamics,
                      parse_full_groups(args.dyn_full_groups)))
            p.start()
            workers.append(p)
        print(f'Launched {args.n_gpus} GPU workers')

        completed = 0
        summaries = []
        while completed < len(groups):
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0 and completed < len(groups):
                print(f'\nAll workers exited after {completed}/{len(groups)} groups')
                break
            try:
                idx, summary = result_queue.get(timeout=1800)
            except Exception:
                continue

            completed += 1
            summaries.append(summary)
            elapsed = (time.time() - start) / 60

            if 'error' in summary:
                print(f'[{completed}/{len(groups)}] '
                      f'n={summary["n"]} m={summary["m"]} S={summary["S"]} '
                      f'GPU{summary.get("gpu_id","?")}: FAIL {summary["error"]} '
                      f'({elapsed:.1f}m)')
            else:
                stages = '  '.join(f'l={l}:{mse:.5f}'
                                   for l, mse in summary['stage_results'].items())
                print(f'[{completed}/{len(groups)}] '
                      f'n={summary["n"]} m={summary["m"]} S={summary["S"]} '
                      f'GPU{summary.get("gpu_id","?")} '
                      f'({summary["elapsed_sec"]/60:.1f}m): {stages} '
                      f'[total {elapsed:.1f}m]')

        for w in workers:
            w.join(timeout=30)

    else:
        # ─── Single-process path ────────────────────────────────────
        torch.manual_seed(args.master_seed)
        np.random.seed(args.master_seed)
        store = ResultsStore(args.store_dir)
        summaries = []
        for i, g in enumerate(groups):
            print(f'\n[{i+1}/{len(groups)}] Group n={g["n"]} m={g["m"]} S={g["S"]} '
                  f'(up to l={g["max_l"]})')
            try:
                summary = fix_group(g['n'], g['m'], g['S'], g['max_l'],
                                    K=args.K,
                                    violated_ls=sorted(set(g['violated_ls'])),
                                    store=store,
                                    batch_size=args.batch_size,
                                    models_dir=os.path.join(args.store_dir, 'models'),
                                    near_warm_start_K=args.near_warm_start_K,
                                    near_warm_start_lr_mult=args.near_warm_start_lr_mult,
                                    explore_K=args.explore_K,
                                    run_id=args.run_id,
                                    master_seed=args.master_seed,
                                    steps_base=args.steps_base,
                                    dyn_dir=(os.path.join(args.store_dir, 'dynamics')
                                             if args.record_dynamics else None),
                                    dyn_full_groups=parse_full_groups(
                                        args.dyn_full_groups))
                summaries.append(summary)
                elapsed = (time.time() - start) / 60
                print(f'  Stage results: ' +
                      '  '.join(f'l={l}:{mse:.5f}' for l, mse in summary['stage_results'].items()))
                print(f'  Total elapsed: {elapsed:.1f}m')
            except Exception as e:
                import traceback
                traceback.print_exc()
                summaries.append(dict(n=g['n'], m=g['m'], S=g['S'], error=str(e)))

    print(f'\nDone in {(time.time()-start)/60:.1f}m')

    # Recompile + report
    store = ResultsStore(args.store_dir)
    store.compile()
    remaining = find_violations(args.store_dir)
    if args.all_groups:
        print(f'Remaining violations: {len(remaining)}')
    else:
        print(f'Remaining violations: {len(remaining)} (was {len(violations)})')

    failed = [s for s in summaries if 'error' in s]
    if failed:
        print(f'FAILED groups: {len(failed)}/{len(groups)}')
        for s in failed[:10]:
            print(f"  n={s['n']} m={s['m']} S={s['S']}: {s['error']}")
        import sys
        sys.exit(1)


if __name__ == '__main__':
    main()
