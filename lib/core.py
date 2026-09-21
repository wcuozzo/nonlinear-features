"""
Core functions for nonlinear feature encoding experiments.
Shared between main notebook and sanity checks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from typing import Dict, List

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')


def _resolve_device(device=None):
    """Return an explicit device, or the module-level default `device` if None.

    Single fallback point so every device-taking function follows ONE
    convention: pass `device=...` to be explicit, or omit it to use the
    module default. Reads the module global at call time, so overriding
    `core.device` still propagates (e.g. forcing CPU in a notebook).
    """
    return globals()['device'] if device is None else device


class Autoencoder(nn.Module):
    """
    Autoencoder with configurable depth.

    l = number of linear layers in encoder:
        l=1: x → Linear(n→m) → z                              [simplest, paper's setup]
        l=2: x → Linear(n→n) → ReLU → Linear(n→m) → z
        l=3: x → Linear(n→n) → ReLU → Linear(n→n) → ReLU → Linear(n→m) → z

    Decoder mirrors encoder structure, with ReLU on final output.
    Default is UNTIED at every depth; pass tied_weights=True for the paper's tied l=1
    (W encode, W.T decode) or the deep-tied l>=2 variant.
    """
    def __init__(self, n: int, m: int, l: int = 1, tied_weights: bool = False, activation=nn.ReLU):
        """
        Args:
            n: feature dimension (input/output width)
            m: bottleneck dimension
            l: number of linear layers in encoder (minimum 1)
            tied_weights: whether the decoder reuses the encoder's weight matrices (transposed).
                Defaults to False = UNTIED at every depth (including l=1). Pass True for the tied
                model: at l=1 the paper's W/W.T tied autoencoder, at l>=2 the deep-tied variant
                (encoder stack + decoder reuses each W.T + own biases). Untied-l1 uses a
                bias-free linear encoder + a free decoder weight (see build note below).
            activation: activation function class (default: ReLU)
        """
        super().__init__()
        assert l >= 1, "l must be at least 1 (need at least one linear layer)"
        self.n = n
        self.m = m
        self.l = l
        # tied_weights ties the weight MATRICES: the decoder reuses the encoder's weight matrices
        # (transposed) instead of learning its own. Defaults to False = UNTIED at every depth.
        # Untying frees the decoder weights; l=1 is then the exactly-linear (crystalline) stratum,
        # compared to the deeper strata on equal footing (no tied-vs-untied confound in the law).
        self.tied_weights = tied_weights

        # Encoder: the SAME feedforward stack for tied AND untied, at every depth —
        # (l-1) hidden [Linear(n→n) + ReLU], then a bias-free bottleneck producer Linear(n→m).
        # The bottleneck layer is BIAS-FREE at every depth: no ReLU follows it, so its bias is
        # redundant (it absorbs exactly into the decoder's first bias); at l=1 dropping it also
        # keeps the encoder positively homogeneous (z(t·x)=t·z(x)). Rule of thumb: a bias only
        # earns its keep right before a ReLU.
        encoder_layers = []
        for i in range(l - 1):
            encoder_layers.append(nn.Linear(n, n))
            encoder_layers.append(activation())
        encoder_layers.append(nn.Linear(n, m, bias=False))  # bottleneck producer: no bias
        self.encoder = nn.Sequential(*encoder_layers)

        if self.tied_weights:
            # Tied: the decoder REUSES each encoder weight matrix (transposed), in reverse order,
            # with its OWN per-layer biases + a ReLU per layer (see decode). This isolates the
            # effect of freeing the decoder weights. l=1 — the Toy Models paper's ReLU(z @ W + b)
            # autoencoder — falls out as the natural l=1 instance, with no special case.
            self.dec_biases = nn.ParameterList(
                [nn.Parameter(torch.zeros(n)) for _ in range(l)])
            self.act = activation()
        else:
            # Untied: an independent decoder stack. Linear(m→n), then (l-1) layers of
            # [ReLU + Linear(n→n)], then a final ReLU so the output is non-negative (matching
            # the sparse non-negative features).
            decoder_layers = [nn.Linear(m, n)]
            for i in range(l - 1):
                decoder_layers.append(activation())
                decoder_layers.append(nn.Linear(n, n))
            decoder_layers.append(activation())  # Final ReLU for non-negative output
            self.decoder = nn.Sequential(*decoder_layers)

        # Weights: Kaiming-uniform, scaled for the activation each layer actually feeds — gain
        # sqrt(2) for the Linears followed by a ReLU, gain 1 for the bias-free bottleneck producer
        # (n->m), which has no activation. Same "is there a ReLU after this?" rule as the biases.
        # This replaces nn.Linear's default kaiming_uniform_(a=sqrt5), whose std sits a factor
        # sqrt(3) below variance-preserving (a legacy accident, not a principled choice).
        # Biases start at ZERO: symmetry-breaking is the weights' job, and a random additive term
        # scaled by 1/sqrt(fan_in) is unjustified (it doesn't sum fan_in inputs).
        encoder_bottleneck = [mod for mod in self.encoder if isinstance(mod, nn.Linear)][-1]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity='linear' if module is encoder_bottleneck else 'relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, x):
        # One path for every architecture (tied and untied, all depths): the encoder Sequential
        # applies its Linear+ReLU stack — ReLU on the hidden layers, none on the bottleneck.
        return self.encoder(x)

    def decode(self, z):
        if not self.tied_weights:
            return self.decoder(z)  # untied: independent decoder stack
        # Tied: walk the encoder's weight matrices in REVERSE, reusing each transposed, with a
        # ReLU + own bias per layer. l=1 reduces to ReLU(z @ W + b) — the Toy Models paper.
        enc_linears = [mod for mod in self.encoder if isinstance(mod, nn.Linear)]
        h = self.act(z @ enc_linears[-1].weight + self.dec_biases[0])          # m -> n
        for idx, k in enumerate(range(self.l - 2, -1, -1), start=1):
            h = self.act(h @ enc_linears[k].weight + self.dec_biases[idx])     # n -> n
        return h

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z



def det_seed(*parts, mod: int = 2**31 - 1) -> int:
    """Deterministic seed from arbitrary (mixed-type) parts.

    Unlike built-in hash(), which is salted per-process for str (PYTHONHASHSEED),
    this is stable across runs, processes, and platforms.
    """
    import hashlib
    key = '|'.join(repr(p) for p in parts)
    return int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) % mod


def generate_sparse_data(n_samples: int, n_features: int, S: float = 0.95, device=None) -> torch.Tensor:
    """
    Generate sparse data where each feature is independently active with probability (1 - S).
    Active features have random positive values.

    Args:
        n_samples: number of samples
        n_features: number of features
        S: sparsity = probability of being ZERO (matches Toy Models paper notation).
           S=0.95 means 5% of features are active on average.
        device: torch device (default: module default `device`)
    """
    device = _resolve_device(device)
    mask = (torch.rand(n_samples, n_features, device=device) > S).float()
    values = torch.rand(n_samples, n_features, device=device)
    return mask * values


def get_feature_importance(n: int, decay: float = 0.7, device=None) -> torch.Tensor:
    """
    Generate importance weights I_i = decay^i (paper uses decay=0.7).

    Args:
        n: number of features
        decay: decay factor (0.7 in Toy Models paper)
        device: torch device (default: module default `device`)

    Returns:
        Tensor of shape (n,) with importance weights
    """
    device = _resolve_device(device)
    return torch.tensor([decay ** i for i in range(n)], device=device, dtype=torch.float32)


def _split_decay_params(model):
    """Split params into (decay, no_decay). Weight matrices get weight decay;
    biases do not (the GPT-2 / Karpathy convention).

    Grouping is by NAME, not ndim. The usual `p.dim() >= 2` test is correct for
    nn.Linear (2-D weight, 1-D bias) but WRONG for BatchedAutoencoder, which
    stores weights as [K, in, out] (3-D) AND biases as [K, 1, n] (3-D) — both
    pass the ndim test the same way. Instead the project maintains a NAMING
    INVARIANT: every bias parameter's name contains the word 'bias'
    (core.Autoencoder: '....bias', 'dec_biases.*'; BatchedAutoencoder:
    'enc_bias.*', 'dec_bias.*', tied 'dec_bias'), and no weight's name does.
    Keep the invariant when adding parameters.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if 'bias' in name else decay).append(p)
    return decay, no_decay


def make_adamw(model, lr, weight_decay=1e-2):
    """AdamW with two shared optimizations, centralized so every training loop
    in the project gets them identically:

      (1) Weight decay on weight matrices only, NOT biases. Biases are
          zero-initialized additive terms; decaying them is a small but real
          confound for a project whose object is a clean scaling law. (The
          bottleneck-producer layers are already bias-free; this handles the
          remaining ReLU-preceding biases.)
      (2) fused=True on CUDA — folds the optimizer step into a single fused
          kernel instead of many small per-parameter launches. This matters
          most for the ParameterList-heavy BatchedAutoencoder. The fused path
          requires every parameter to live on CUDA, so we gate on the model's
          actual parameter device (not a passed-in device string), which also
          makes CPU/MPS fall back to the standard path automatically.
    """
    decay, no_decay = _split_decay_params(model)
    groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ]
    use_fused = next(model.parameters()).is_cuda  # param-less model: let it crash (AdamW would anyway)
    return optim.AdamW(groups, lr=lr, fused=use_fused)


def train_autoencoder(
    model: Autoencoder,
    n_steps: int = 10000,
    batch_size: int = 1024,
    S: float = 0.95,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    importance: torch.Tensor = None,
    loss_threshold: float = None,
    verbose: bool = True,
    device=None
) -> List[float]:
    """
    Train the autoencoder on sparse data.

    Args:
        model: Autoencoder to train
        n_steps: number of training steps
        batch_size: batch size
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
        lr: learning rate
        weight_decay: L2 regularization (Adam weight_decay parameter)
        importance: optional feature importance weights I_i (shape: n)
        loss_threshold: if set, stop training early when loss drops below this value
        final_loss_window: how many trailing training-batch losses to average into
            final_loss (a convergence-tail estimate; NOT a precise eval — the canon
            re-evaluates saved models at n=200k in precise_eval)
        verbose: print progress
        device: torch device for generated data (default: module default `device`).
                Should match the device the model is on.

    Returns:
        List of loss values per step
    """
    device = _resolve_device(device)
    # Paper uses AdamW (decoupled weight decay), not Adam with L2 regularization.
    # make_adamw adds fused-on-CUDA + no-weight-decay-on-biases.
    optimizer = make_adamw(model, lr=lr, weight_decay=weight_decay)
    losses = []

    # Reading loss.item() every step forces a CPU<->GPU sync that serializes the
    # GPU (the accelerator stalls waiting for the scalar). We keep the per-step
    # loss on-device and drain to the host in one transfer every FLUSH steps, so
    # the returned per-step list is unchanged but we pay ~1 sync per FLUSH steps
    # instead of one per step. The verbose readout and threshold early-stop opt
    # back into a scalar read only on the (rare) steps they actually fire.
    FLUSH = 1000
    loss_buffer = []  # detached on-device loss scalars, flushed in batches

    iterator = tqdm(range(n_steps)) if verbose else range(n_steps)
    for step in iterator:
        x = generate_sparse_data(batch_size, model.n, S, device=device)

        optimizer.zero_grad()
        x_recon, z = model(x)

        if importance is not None:
            # Weighted MSE loss: mean over batch of (importance * (x - x_recon)^2)
            loss = (importance * (x - x_recon) ** 2).mean()
        else:
            loss = nn.functional.mse_loss(x_recon, x)

        loss.backward()
        optimizer.step()

        loss_buffer.append(loss.detach())
        if len(loss_buffer) >= FLUSH:
            losses.extend(torch.stack(loss_buffer).cpu().tolist())
            loss_buffer.clear()

        if verbose and step % 1000 == 0:
            iterator.set_postfix({'loss': f'{loss.detach().item():.6f}'})

        # Early stopping (opts into a per-step sync only when a threshold is set)
        if loss_threshold is not None:
            cur = loss.detach().item()
            if cur < loss_threshold:
                if verbose:
                    print(f"Early stop at step {step}, loss={cur:.6f}")
                break

    if loss_buffer:
        losses.extend(torch.stack(loss_buffer).cpu().tolist())

    return losses


def measure_hidden_state_linearity(model: Autoencoder, n_samples: int = 1000, S: float = 0.95, device=None) -> float:
    """Hidden-state linearity: R^2 of the best affine fit x -> z.

    1.0 = the encoding is exactly affine (true at l=1 by construction);
    lower = genuinely nonlinear encoding. Fit by least squares on fresh
    sparse data. Measures ONLY this — loss is reported elsewhere
    (final_loss in run_experiment; precise_eval for the canon).

    Args:
        model: trained autoencoder
        n_samples: number of samples for evaluation
        S: sparsity = probability of being ZERO (matches Toy Models paper notation)
        device: torch device (default: module default `device`). Should match the model.
    """
    device = _resolve_device(device)
    model.eval()

    with torch.no_grad():
        x = generate_sparse_data(n_samples, model.n, S, device=device)
        z = model.encode(x)

        x_with_bias = torch.cat([x, torch.ones(n_samples, 1, device=device)], dim=1)
        W_linear = torch.linalg.lstsq(x_with_bias, z).solution
        z_linear = x_with_bias @ W_linear

        z_var = z.var(dim=0).sum()
        residual_var = (z - z_linear).var(dim=0).sum()

    return 1 - (residual_var / z_var).item()


# Two-dataset protocol: seeds are SELECTED on the selection draw, then the winner is
# REPORTED/COMPARED on the canonical test draw (seed 42, shared with precise_eval).
# Selecting and reporting on the same draw would bias the winner's score low (it was
# picked partly for luck on that data — winner's curse). Production mirrors this:
# in-run selection at seed 99999, canonical re-eval at seed 42.
SELECTION_EVAL_SEED = 99999


def evaluate_mse(model: Autoencoder, S: float, n_samples: int = 200_000,
                 seed: int = 42, device=None) -> Dict[str, float]:
    """Precise PAIRED eval: final fixed weights, fixed data seed.

    The canonical fixed draw (seed=42, n=200k, chunk=50k) — precise_eval.py applies THIS fn,
    so numbers line up with the canonical CSV, and the shared seed makes comparisons
    paired (common random numbers) across seeds AND across configs at the same (n, S)
    — data-draw luck cancels out of comparisons instead of adding noise to them.

    Returns {'eval_mse', 'eval_mse_ci95'}: ci95 = 1.96 * SE of the per-sample mean
    squared error — the "eval luck" scale. At n=200k it is a small fraction of mse;
    differences within ~a ci95 of each other are not resolved by this eval.
    """
    device = _resolve_device(device)
    model.eval()
    torch.manual_seed(seed)
    per_sample = []
    with torch.no_grad():
        chunk = 50000                        # matches precise_mse for identical draws
        for start in range(0, n_samples, chunk):
            cs = min(chunk, n_samples - start)
            x = generate_sparse_data(cs, model.n, S, device=device)
            x_hat, _ = model(x)
            per_sample.append(((x_hat - x) ** 2).mean(dim=1))   # per-sample MSE
    ps = torch.cat(per_sample)
    model.train()
    return {
        'eval_mse': ps.mean().item(),
        'eval_mse_ci95': (1.96 * ps.std() / (len(ps) ** 0.5)).item(),
    }


def run_experiment(
    n: int, m: int, l: int = 1,
    S: float = 0.95,
    n_steps: int = 10000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    importance_decay: float = None,
    tied_weights: bool = False,
    loss_threshold: float = None,
    final_loss_window: int = 100,
    verbose: bool = True,
    device=None
) -> Dict:
    """
    Run a single experiment with given parameters.

    Args:
        n: input/output dimension (number of features)
        m: bottleneck dimension
        l: number of linear layers in encoder (l=1 is simplest, matches paper)
        S: sparsity = probability of being ZERO (matches Toy Models paper notation).
           S=0.95 means 5% of features are active on average.
        n_steps: training steps
        batch_size: batch size (paper uses 1024)
        lr: learning rate
        weight_decay: L2 regularization (default 1e-2 matches paper)
        importance_decay: if set, use I_i = importance_decay^i weighting (paper uses 0.7)
        tied_weights: decoder reuses encoder weights (transposed). Default False = untied
            at every depth, matching the canonical sweep. Pass True explicitly to replicate
            the paper's tied setup (e.g. l=1, tied_weights=True).
        loss_threshold: if set, stop training early when loss drops below this value
        final_loss_window: how many trailing training-batch losses to average into
            final_loss (a convergence-tail estimate; NOT a precise eval — the canon
            re-evaluates saved models at n=200k in precise_eval)
        verbose: print progress
        device: torch device (default: module default `device`)

    Returns:
        Dict with model, metrics, and training info
    """
    device = _resolve_device(device)
    model = Autoencoder(n, m, l, tied_weights=tied_weights).to(device)

    # Compute importance weights if specified
    importance = None
    if importance_decay is not None:
        importance = get_feature_importance(n, importance_decay, device=device)

    if verbose:
        print(f"\nExperiment: n={n}, m={m}, l={l}, S={S}")
        if importance_decay is not None:
            print(f"Using importance weighting: I_i = {importance_decay}^i")
        print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")

    losses = train_autoencoder(
        model, n_steps=n_steps, batch_size=batch_size, S=S,
        lr=lr, weight_decay=weight_decay, importance=importance,
        loss_threshold=loss_threshold,
        verbose=verbose, device=device
    )

    hidden_state_linearity = measure_hidden_state_linearity(model, S=S, device=device)
    eval_metrics = evaluate_mse(model, S=S, device=device)

    results = {
        'n': n, 'm': m, 'l': l, 'S': S,
        'importance_decay': importance_decay,
        'final_loss': np.mean(losses[-final_loss_window:]) if len(losses) >= final_loss_window else np.mean(losses),
        **eval_metrics,
        'hidden_state_linearity': hidden_state_linearity,
        'model': model,
        'losses': losses,
    }

    if verbose:
        print(f"Results: hidden_state_linearity={hidden_state_linearity:.3f}")

    return results


def run_experiment_multi_seed(
    n: int, m: int, l: int = 1,
    n_seeds: int = 10,
    S: float = 0.95,
    n_steps: int = 10000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    importance_decay: float = None,
    tied_weights: bool = False,
    loss_threshold: float = None,
    verbose: bool = True,
    device=None
) -> Dict:
    """
    Run multiple seeds and return the best (lowest loss) result.

    The Toy Models paper runs 200+ seeds due to optimization sensitivity.
    This function runs n_seeds experiments and keeps the best one.

    Args:
        n: input/output dimension
        m: bottleneck dimension
        l: number of linear layers in encoder (l=1 is simplest, matches paper)
        n_seeds: number of random seeds to try
        S: sparsity = probability of being ZERO (matches Toy Models paper notation).
           S=0.95 means 5% of features are active on average.
        n_steps: training steps per seed
        batch_size: batch size (paper uses 1024)
        lr: learning rate
        weight_decay: L2 regularization (default 1e-2 matches paper)
        importance_decay: if set, use I_i = importance_decay^i weighting
        tied_weights: decoder reuses encoder weights (transposed). Default False = untied
            at every depth; pass True explicitly for the paper's tied setup.
        loss_threshold: if set, stop searching seeds once best loss is below this.
            (Geometry-based acceptance gates — min/max norm, min angle — were removed:
            that is pentagon-reproduction logic and belongs in the dedicated TMS
            reproduction notebook, not the generic multi-seed search.)
        verbose: print progress
        device: torch device (default: module default `device`)

    Returns:
        Dict with best model, all losses, and seed info
    """
    device = _resolve_device(device)
    best_result = None
    best_loss = float('inf')
    best_seed = None
    all_selection_mses = []

    if verbose:
        print(f"\nMulti-seed experiment: n={n}, m={m}, l={l}, S={S}, n_seeds={n_seeds}")
        if importance_decay is not None:
            print(f"Using importance weighting: I_i = {importance_decay}^i")
        if loss_threshold is not None:
            print(f"Early stopping: loss<{loss_threshold}")

    iterator = tqdm(range(n_seeds), desc="Seeds") if verbose else range(n_seeds)

    for seed in iterator:
        torch.manual_seed(seed)
        np.random.seed(seed)

        result = run_experiment(
            n=n, m=m, l=l,
            S=S,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            importance_decay=importance_decay,
            tied_weights=tied_weights,
            loss_threshold=None,  # Don't stop training early - let it converge fully
            verbose=False,
            device=device
        )

        # Select on a PAIRED precise eval (final weights, common data seed), but on
        # the SELECTION draw — NOT the seed-42 test draw that run_experiment attaches
        # and that cross-config comparisons use. Selecting and comparing on the same
        # data would select for luck on it (winner's curse); the winner's reported
        # eval_mse stays untouched by selection this way.
        eval_loss = evaluate_mse(result['model'], S=S,
                                 seed=SELECTION_EVAL_SEED, device=device)['eval_mse']
        all_selection_mses.append(eval_loss)

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_result = result
            best_seed = seed

        if verbose:
            iterator.set_postfix({
                'loss': f'{eval_loss:.4f}'
            })

        # Stop searching seeds once the best is good enough.
        if loss_threshold is not None and best_loss < loss_threshold:
            break

    # Add multi-seed info to result
    best_result['best_seed'] = best_seed
    best_result['selection_mse'] = best_loss          # winner's score on the SELECTION draw
    best_result['all_selection_mses'] = all_selection_mses
    best_result['n_seeds'] = n_seeds
    best_result['seeds_tried'] = len(all_selection_mses)


    if verbose:
        print(f"\nBest seed: {best_seed}, selection_mse: {best_loss:.6f}, test eval_mse: {best_result['eval_mse']:.6f}")
        print(f"Seeds tried: {len(all_selection_mses)}/{n_seeds}")
        print(f"Loss range across seeds: [{min(all_selection_mses):.6f}, {max(all_selection_mses):.6f}]")
        print(f"Results: hidden_state_linearity={best_result['hidden_state_linearity']:.3f}")

    return best_result

