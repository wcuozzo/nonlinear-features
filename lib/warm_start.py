"""
Multi-source warm-start utility for autoencoder retraining.

Given a target config (n, m, l, S) that we want to retrain, this module
constructs candidate initializations by combining multiple sources:

  1. SHALLOW: best converged (n, m, l', S) with l' < l, embedded with
     identity middle layers. (data is non-negative => identity + ReLU is
     a true pass-through). This is the canonical warm start.
  2. EXISTING: best existing (n, m, l, S) model, perturbed slightly.
  3. NEIGHBOR: best (n, m', l, S) with m' < m, projected to bottleneck m
     by zero-padding or random-padding the bottleneck.
  4. RANDOM: a few fresh random seeds for diversity.

Each candidate is verified to give MSE <= shallow_floor before being
accepted. If verification fails, the candidate is reinitialized with
less noise or dropped.
"""

import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from core import Autoencoder, generate_sparse_data
import core


# ────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────

def get_linear_layers(part):
    return [c for c in part.children() if isinstance(c, nn.Linear)]


def eval_mse(model: Autoencoder, S: float, n_samples: int = 8000,
             seed: int = 99999, device=None) -> float:
    """Deterministic MSE measurement on a held-out fixed sample."""
    dev = device or core.device
    model.eval()
    torch.manual_seed(seed)
    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S)
        x_hat, _ = model(x)
        mse = ((x_hat - x) ** 2).mean().item()
    return mse


def load_best_model(n: int, m: int, l: int, S: float,
                    models_dir: str = 'data_and_models/results_db/models',
                    device=None) -> Optional[Autoencoder]:
    """Load a converged model from the results store, if present."""
    path = Path(models_dir) / f'model_n{n}_m{m}_l{l}_S{S}.pt'
    if not path.exists():
        return None
    dev = device or core.device
    model = Autoencoder(n, m, l, tied_weights=False).to(dev)
    model.load_state_dict(torch.load(path, map_location=dev))
    return model


# ────────────────────────────────────────────────────────────────────────
# Source 1: SHALLOW embed
# ────────────────────────────────────────────────────────────────────────

def embed_shallow_in_deep(shallow_model: Autoencoder, deep_l: int,
                          noise: float = 0.0, seed: int = 0,
                          device=None) -> Autoencoder:
    """Embed a converged shallow model in a deeper architecture with
    identity middle layers.

    Works because:
      - data is non-negative (each feature is mask * Uniform[0,1])
      - identity + ReLU on non-negative input is the identity
      - decoder's final ReLU output is non-negative, so subsequent
        identity layers + ReLU preserve it

    With noise=0 this gives EXACTLY shallow's MSE (up to floating point).
    """
    dev = device or core.device
    n, m = shallow_model.n, shallow_model.m
    l_shallow = shallow_model.l
    assert deep_l > l_shallow
    deep = Autoencoder(n, m, deep_l, tied_weights=False).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)

    deep_enc = get_linear_layers(deep.encoder)
    deep_dec = get_linear_layers(deep.decoder)

    with torch.no_grad():
        # ─── ENCODER: identity layers, then shallow's encoder ───
        # The encoder is a Sequential at every depth (tied and untied), so the shallow encoder's
        # Linears extract uniformly: all but the last are "middle" layers to copy, the last is the
        # bottleneck producer. (At l_shallow=1 there are simply no middle layers.)
        sl = get_linear_layers(shallow_model.encoder)
        shallow_enc_middle = sl[:-1]
        shallow_enc_comp = sl[-1]

        n_ident_enc = deep_l - l_shallow
        for i in range(n_ident_enc):
            nn.init.eye_(deep_enc[i].weight)
            if noise > 0:
                deep_enc[i].weight.add_(
                    torch.randn(deep_enc[i].weight.shape,
                                generator=gen, device=dev) * noise)
            deep_enc[i].bias.zero_()
        for j, layer in enumerate(shallow_enc_middle):
            deep_enc[n_ident_enc + j].weight.copy_(layer.weight)
            deep_enc[n_ident_enc + j].bias.copy_(layer.bias)
        deep_enc[-1].weight.copy_(shallow_enc_comp.weight)
        # deep_enc[-1] is the bottleneck layer: bias-free at every depth, nothing to copy.

        # ─── DECODER: shallow's decoder, then identity layers ───
        if shallow_model.tied_weights and l_shallow == 1:
            # Tied l=1 (TMS anchor): decode is ReLU(z @ enc.weight + dec_bias)
            #   deep_dec[0] is Linear(m,n); out = z @ W.T + b => W = enc.weight.T
            enc_W = get_linear_layers(shallow_model.encoder)[-1].weight
            deep_dec[0].weight.copy_(enc_W.T)
            deep_dec[0].bias.copy_(shallow_model.dec_biases[0])
            shallow_dec_middle = []
        else:
            # Untied (any depth, incl. l=1): shallow has a real decoder stack.
            sl = get_linear_layers(shallow_model.decoder)
            deep_dec[0].weight.copy_(sl[0].weight)
            deep_dec[0].bias.copy_(sl[0].bias)
            shallow_dec_middle = sl[1:]

        for j, layer in enumerate(shallow_dec_middle):
            deep_dec[1 + j].weight.copy_(layer.weight)
            deep_dec[1 + j].bias.copy_(layer.bias)

        n_ident_dec = deep_l - l_shallow
        start = 1 + len(shallow_dec_middle)
        for i in range(start, start + n_ident_dec):
            nn.init.eye_(deep_dec[i].weight)
            if noise > 0:
                deep_dec[i].weight.add_(
                    torch.randn(deep_dec[i].weight.shape,
                                generator=gen, device=dev) * noise)
            deep_dec[i].bias.zero_()

    return deep


# ────────────────────────────────────────────────────────────────────────
# Source 2: EXISTING model perturbation
# ────────────────────────────────────────────────────────────────────────

def perturb_model(model: Autoencoder, noise: float, seed: int = 0,
                  device=None) -> Autoencoder:
    """Clone a model and add Gaussian noise to each parameter."""
    dev = device or core.device
    new = Autoencoder(model.n, model.m, model.l,
                      tied_weights=model.tied_weights).to(dev)
    new.load_state_dict(model.state_dict())
    gen = torch.Generator(device=dev).manual_seed(seed)
    with torch.no_grad():
        for p in new.parameters():
            p.add_(torch.randn(p.shape, generator=gen, device=dev) * noise)
    return new


# ────────────────────────────────────────────────────────────────────────
# Source: S-NEIGHBOR — same (n, m, l), adjacent sparsity
# ────────────────────────────────────────────────────────────────────────

def load_S_neighbor(n: int, m: int, l: int, S: float, S_values,
                    models_dir: str = 'data_and_models/results_db/models',
                    device=None):
    """Best model at the SAME (n, m, l) but a neighboring S. Identical
    architecture, so the weights load completely unchanged — the cheapest
    warm-start source of all. Prefers the nearest S (denser first: a model
    that handles more simultaneous activity transfers well to sparser data).
    Returns (model, S_source) or (None, None)."""
    others = sorted((s2 for s2 in S_values if s2 != S), key=lambda s2: (abs(s2 - S), s2))
    for S2 in others:
        cand = load_best_model(n, m, l, S2, models_dir, device=device)
        if cand is not None and not cand.tied_weights:
            return cand, S2
    return None, None


# ────────────────────────────────────────────────────────────────────────
# Source 3: NEIGHBOR config (smaller bottleneck) projection
# ────────────────────────────────────────────────────────────────────────

def embed_narrower_bottleneck(narrow_model: Autoencoder, target_m: int,
                              seed: int = 0, device=None) -> Autoencoder:
    """Take a model with bottleneck m_narrow < target_m and embed it
    in an architecture with target_m.

    The first m_narrow bottleneck dims get the trained values; the
    remaining (target_m - m_narrow) dims get small-random weights.
    """
    dev = device or core.device
    n, m_narrow, l = narrow_model.n, narrow_model.m, narrow_model.l
    assert target_m > m_narrow
    assert not narrow_model.tied_weights  # only support l>=2 here
    new = Autoencoder(n, target_m, l, tied_weights=False).to(dev)
    gen = torch.Generator(device=dev).manual_seed(seed)

    with torch.no_grad():
        nl = get_linear_layers(narrow_model.encoder)
        new_enc = get_linear_layers(new.encoder)
        # All n->n encoder layers: identical
        for i in range(len(nl) - 1):
            new_enc[i].weight.copy_(nl[i].weight)
            new_enc[i].bias.copy_(nl[i].bias)
        # Last (compression) layer: Linear(n, m_narrow) -> Linear(n, target_m)
        # weight shape [m, n]; copy first m_narrow rows from narrow, randomize rest
        new_enc[-1].weight.zero_()
        new_enc[-1].weight[:m_narrow].copy_(nl[-1].weight)
        new_enc[-1].weight[m_narrow:] = (
            torch.randn(target_m - m_narrow, n,
                        generator=gen, device=dev) * 0.01)
        # bottleneck (compression) layer is bias-free at every depth — no bias to project

        nl = get_linear_layers(narrow_model.decoder)
        new_dec = get_linear_layers(new.decoder)
        # First (expansion) layer: Linear(m_narrow, n) -> Linear(target_m, n)
        # weight shape [n, m]; first m_narrow cols from narrow, rest zero
        new_dec[0].weight.zero_()
        new_dec[0].weight[:, :m_narrow].copy_(nl[0].weight)
        new_dec[0].bias.copy_(nl[0].bias)
        for i in range(1, len(nl)):
            new_dec[i].weight.copy_(nl[i].weight)
            new_dec[i].bias.copy_(nl[i].bias)

    return new


# ────────────────────────────────────────────────────────────────────────
# Source 4: RANDOM
# ────────────────────────────────────────────────────────────────────────

def random_init(n: int, m: int, l: int, seed: int, device=None) -> Autoencoder:
    dev = device or core.device
    torch.manual_seed(seed)
    return Autoencoder(n, m, l, tied_weights=False).to(dev)


# ────────────────────────────────────────────────────────────────────────
# Multi-source init builder
# ────────────────────────────────────────────────────────────────────────

def build_init_pool(n: int, m: int, l: int, S: float, K: int,
                    models_dir: str = 'data_and_models/results_db/models',
                    base_seed: int = 1000,
                    n_random: int = 2,
                    perturbation_noise: float = 0.01,
                    identity_noise: float = 0.0,
                    floor_tolerance: float = 1.10,
                    device=None,
                    verbose: bool = False) -> List[dict]:
    """Build K initializations from multiple sources.

    Returns a list of K dicts, each with:
        'model': nn.Module ready to train
        'source': one of {'shallow', 'existing', 'neighbor', 'random'}
        'init_mse': measured floor MSE
        'shallow_mse': the shallow MSE we should beat
        'description': human-readable
    """
    dev = device or core.device
    pool = []

    # ── Pre-compute shallow floor MSE for reference ──
    shallow_mse = float('inf')
    best_shallow_l = None
    best_shallow = None
    for l_s in range(1, l):
        sh = load_best_model(n, m, l_s, S, models_dir, dev)
        if sh is None:
            continue
        mse_s = eval_mse(sh, S, device=dev)
        if mse_s < shallow_mse:
            shallow_mse = mse_s
            best_shallow_l = l_s
            best_shallow = sh
    floor_mse = shallow_mse * floor_tolerance

    if verbose:
        print(f'  Shallow floor: l={best_shallow_l}, MSE={shallow_mse:.6f}')

    # ── Source 1: SHALLOW embeds (different noise levels) ──
    n_shallow_target = max(K // 2, 1) if best_shallow is not None else 0
    noise_options = [0.0, 0.001, 0.003, identity_noise]
    if best_shallow is not None:
        added = 0
        for i in range(n_shallow_target * 2):  # try more to handle skips
            if added >= n_shallow_target:
                break
            noise = noise_options[i % len(noise_options)]
            seed = base_seed + i
            init = embed_shallow_in_deep(best_shallow, l,
                                         noise=noise, seed=seed, device=dev)
            mse = eval_mse(init, S, device=dev)
            if mse <= floor_mse:
                pool.append(dict(
                    model=init, source='shallow', init_mse=mse,
                    shallow_mse=shallow_mse,
                    description=f'shallow(l={best_shallow_l})+noise={noise}',
                ))
                added += 1
            elif verbose:
                print(f'  [skip] shallow noise={noise} seed={seed}: '
                      f'init_mse={mse:.5f} > {floor_mse:.5f}')

    # ── Source 2: EXISTING (n, m, l) perturbed ──
    existing = load_best_model(n, m, l, S, models_dir, dev)
    if existing is not None:
        mse_existing = eval_mse(existing, S, device=dev)
        # Only worth using if it's already below floor (not the cause of violation)
        if mse_existing <= floor_mse * 2.0:
            for i in range(2):
                seed = base_seed + 100 + i
                init = perturb_model(existing, noise=perturbation_noise,
                                     seed=seed, device=dev)
                mse = eval_mse(init, S, device=dev)
                if mse <= floor_mse * 1.5:
                    pool.append(dict(
                        model=init, source='existing', init_mse=mse,
                        shallow_mse=shallow_mse,
                        description=f'existing+noise={perturbation_noise}',
                    ))

    # ── Source 3: NEIGHBOR (n, m_narrow, l) ──
    # Pick the widest narrower m that has a converged model at same l
    for m_narrow in sorted([2, 4, 8, 16, 32, 64], reverse=True):
        if m_narrow >= m:
            continue
        narrow = load_best_model(n, m_narrow, l, S, models_dir, dev)
        if narrow is None or narrow.tied_weights:
            continue
        for i in range(2):
            seed = base_seed + 200 + i
            init = embed_narrower_bottleneck(narrow, m, seed=seed, device=dev)
            mse = eval_mse(init, S, device=dev)
            if mse <= floor_mse * 2.0:
                pool.append(dict(
                    model=init, source='neighbor', init_mse=mse,
                    shallow_mse=shallow_mse,
                    description=f'neighbor(m={m_narrow})',
                ))
        break  # only use the closest neighbor

    # ── Source 4: RANDOM ──
    for i in range(n_random):
        seed = base_seed + 300 + i
        init = random_init(n, m, l, seed=seed, device=dev)
        mse = eval_mse(init, S, device=dev)
        pool.append(dict(
            model=init, source='random', init_mse=mse,
            shallow_mse=shallow_mse,
            description=f'random(seed={seed})',
        ))

    # ── Pad up to K with extra shallow embeds at noise=0 ──
    while len(pool) < K and best_shallow is not None:
        seed = base_seed + 500 + len(pool)
        init = embed_shallow_in_deep(best_shallow, l, noise=0.0,
                                     seed=seed, device=dev)
        mse = eval_mse(init, S, device=dev)
        pool.append(dict(
            model=init, source='shallow', init_mse=mse,
            shallow_mse=shallow_mse,
            description=f'shallow(l={best_shallow_l})+noise=0[extra]',
        ))

    return pool[:K]
