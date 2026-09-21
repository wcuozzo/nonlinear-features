# Reproducing the results

The README presents two results: nonlinear representations across encoder depths,
and the mechanisms enabled by untied read and write weights. Saved outputs are
included, so reading the results does not require a GPU or retraining.

## Install

Use Python 3.11 or newer; the standalone snapshot was checked with Python 3.13.5.
From this repository's root:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r results/requirements.txt
```

The requirements pin the versions used for the local checks. GPU-enabled PyTorch
installation may require a platform-specific wheel. The numerical checks and
figure regeneration below use CPU.

## Check the saved results

```sh
python scripts/verify_results.py
python -m unittest discover -s src/sweep -p 'test*.py'
python -m unittest discover -s scripts -p 'test_reproduction.py'
```

The first command loads all 216 selected checkpoints, checks their shapes and
finite weights, refits the displayed scaling law, and evaluates the pictured
tied and untied models on their recorded independent test draw. It performs no
training and needs no network access. The second runs the sweep audit and launcher
regression checks without running a training sweep.

The third command checks verbose training, executes the pentagon notebook with a
tiny training budget, and exports the shallow-model diagrams from saved weights.
The reduced notebook run checks execution, not convergence or research results.

`src/sweep/check_results.py` audits new training stores and requires their seed
logs. For the bundled snapshot, use `scripts/verify_results.py` as shown above;
the training auditor stops without writing files when those logs are missing.

## Regenerate the main representation figures

```sh
python results/core/nonlinear_representations/canonical_analysis.py
python results/core/nonlinear_representations/architecture_figure.py
```

The first command regenerates the heatmap, scaling fit, feature trajectories,
speeds, and decoder-region figures under `figures/`, together with their
`canonical_analysis.json` provenance record. It overwrites those generated files
in this checkout. Pixel-level differences can occur across plotting environments.

## Regenerate the shallow-model diagrams

```sh
python scripts/export_shallow_figures.py
```

This recreates the five-feature read/write diagram, its three mechanism panels,
and the pentagon diagram from the included weights, without retraining. It writes
the five corresponding PNGs under `figures/`. To preview them elsewhere, pass
`--output-dir /tmp/shallow-figures`. The exporter preserves the saved geometry and
computes the displayed values from the weights; styling can differ slightly from
the original exported images.

## Source notebooks for the shallow examples

```sh
python -m jupyter lab
```

- [How untying helps](core/tied_untied/how_untying_helps.ipynb): training setup for the five-feature untied example.
- [Toy Models replication](core/tied_untied/toy_models_sanity_check.ipynb): training and geometry checks for the tied pentagon example.

These are the only notebooks included. The main representation results are
produced by the training and analysis scripts described above. The
[weight-decay sweep](core/tied_untied/weight_decay_results/README.md) and
[unit-norm searches](core/tied_untied/one_sided_norm_results/REPORT.md)
also use standalone scripts.

Notebook outputs are retained. Executing the notebooks trains new models and can
take substantial time. Initialization cells locate this checkout before importing
shared code, so notebooks can be opened from their own folders or the repository root.

## Which data belong to which result?

| Files | Role |
|---|---|
| `data_and_models/results_canonical_sweep_2026-08-18_seed42/` | The 216-row all-untied sweep and matching checkpoints used by the main README figures. |
| `figures/untied_read_write_example.json` | Weights and evaluation details for the pictured five-feature example. |
| `results/core/tied_untied/*_results/` | Saved regularization and normalization searches. |
| `results/exploratory/seed_models/` | Saved model for the pentagon reproduction. |

The saved CSV and checkpoints reproduce the reported analysis. The included training
code is the current implementation of the described procedure, with later revisions.
It is intended to reproduce comparable results; a new run can differ in learned
weights, individual losses, and fitted coefficients.

The shallow examples and regularization searches are numerical solutions, not
proofs of global optimality. Their sampling intervals do not measure uncertainty
over all possible optimization runs. `results/ARTIFACTS.json` records hashes for the
included research artifacts.

## Train a new sweep

The full sweep is substantial. Choose a new output directory to avoid overwriting
the supplied data:

```sh
STORE_DIR=new-runs/my-sweep MASTER_SEED=42 N_GPUS=1 bash src/sweep/canonical_sweep.sh
```

For the script's smaller end-to-end smoke configuration:

```sh
SMOKE=1 STORE_DIR=new-runs/smoke N_GPUS=1 DEVICE=cpu bash src/sweep/canonical_sweep.sh
```

The smoke run checks the training workflow; its losses are not research results.
Its retraining stage uses 4 diverse initializations plus up to 2 near-warm-start
initializations, with a 1,500-step base budget. `RESOLVE_K` and
`RESOLVE_STEPS_BASE` override the diverse-initialization count and step budget.
To analyze a new completed run, change `store` in
[canonical_source.json](core/nonlinear_representations/canonical_source.json)
and rerun the analysis after checking the new run with `src/sweep/check_results.py`.
