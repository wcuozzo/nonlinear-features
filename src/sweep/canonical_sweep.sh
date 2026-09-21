#!/bin/bash
# Single canonical sweep into a fresh, dated, self-contained results directory.
# Every (n, m, l, S) trained from scratch with the same recipe (near-warm-start
# arm enabled) and tagged with one master seed + one run_id. Never reads from or
# writes to any pre-existing store (data_and_models/results_db is untouched).
#
# Reproduction and analysis notes: results/REPRODUCING.md
#
# Usage:
#   bash src/sweep/canonical_sweep.sh                   # default seed=42, dated dir
#   MASTER_SEED=7 bash src/sweep/canonical_sweep.sh     # custom seed
#   STORE_DIR=my_run/ bash src/sweep/canonical_sweep.sh # custom output dir
#   SMOKE=1 bash src/sweep/canonical_sweep.sh           # tiny n=16 slice, reduced steps/K —
#                                          # validates the pipeline end-to-end
#                                          # in minutes (NOT scientifically
#                                          # meaningful MSEs)
#
# Output: $STORE_DIR/  with models/ + seeds/ + compiled/sweep_results_precise.csv
#         + manifest.json + sweep.log

set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

# Determinism: seed derivation no longer uses built-in hash(), but pin the hash
# salt anyway so any remaining stragglers are stable too.
export PYTHONHASHSEED=0
# MPS lacks batched linalg_lstsq — fall back to CPU for that op (no-op on CUDA).
export PYTORCH_ENABLE_MPS_FALLBACK=1

DATE_TAG="$(date +%Y-%m-%d)"
MASTER_SEED="${MASTER_SEED:-42}"
SMOKE="${SMOKE:-0}"
PY="${PY:-python3}"

if [ "$SMOKE" = "1" ]; then
  STORE_DIR="${STORE_DIR:-data_and_models/results_smoke_${DATE_TAG}_seed${MASTER_SEED}}"
  RUN_ID="smoke_${DATE_TAG}_seed${MASTER_SEED}"
  N_GPUS=${N_GPUS:-1}
  K=${K:-3}
  NWS_K=${NWS_K:-2}
  BATCH_SIZE=${BATCH_SIZE:-2048}
  CHAIN_STEPS_BASE=${CHAIN_STEPS_BASE:-1500}
  RESOLVE_K=${RESOLVE_K:-4}
  RESOLVE_STEPS_BASE=${RESOLVE_STEPS_BASE:-1500}
  SWEEP_NS="${SWEEP_NS:-16}"
  SWEEP_MS="${SWEEP_MS:-2,4,8}"
  SWEEP_SS="${SWEEP_SS:-0.85,0.9,0.95}"
else
  STORE_DIR="${STORE_DIR:-data_and_models/results_canonical_sweep_${DATE_TAG}_seed${MASTER_SEED}}"
  RUN_ID="canonical_${DATE_TAG}_seed${MASTER_SEED}"
  N_GPUS=${N_GPUS:-8}
  K=${K:-10}
  NWS_K=${NWS_K:-5}
  BATCH_SIZE=${BATCH_SIZE:-8192}
  CHAIN_STEPS_BASE=${CHAIN_STEPS_BASE:-12000}
  RESOLVE_K=${RESOLVE_K:-30}
  RESOLVE_STEPS_BASE=${RESOLVE_STEPS_BASE:-24000}
  SWEEP_NS="${SWEEP_NS:-16,32,64,128}"
  SWEEP_MS="${SWEEP_MS:-2,4,8,16,32,64}"
  SWEEP_SS="${SWEEP_SS:-0.85,0.9,0.95}"
fi

NWS_LR_MULT=${NWS_LR_MULT:-0.3}

# Training-dynamics traces: W_eff/loss at ~40 log-spaced steps for EVERY
# config/seed/arm (fp16, RNG-neutral). Groups in DYN_FULL_GROUPS also get
# full weight snapshots (for LLC/Hessian work). Set RECORD_DYNAMICS=0 to skip.
RECORD_DYNAMICS=${RECORD_DYNAMICS:-1}
DYN_FULL_GROUPS="${DYN_FULL_GROUPS:-16,2,0.9 64,8,0.9}"
DYN_ARGS=""
if [ "$RECORD_DYNAMICS" = "1" ]; then
  DYN_ARGS="--record-dynamics --dyn-full-groups $DYN_FULL_GROUPS"
fi

# Auto-detect eval device unless given (cuda:0 > mps > cpu)
DEVICE="${DEVICE:-$($PY -c "import torch; print('cuda:0' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')")}"

# ── Protect existing data: refuse to write into a non-empty store ──
# (RESUME=1 continues a crashed run in the SAME store: completed groups/configs
#  are skipped — sound because they are deterministic and independent. Invoke
#  with the same env vars as the original run.)
RESUME=${RESUME:-0}
RESUME_ARGS=""
if [ -e "$STORE_DIR/seeds" ] || [ -e "$STORE_DIR/models" ]; then
  if [ "$RESUME" = "1" ] && [ -f "$STORE_DIR/manifest.json" ]; then
    RUN_ID=$($PY -c "import json; print(json.load(open('$STORE_DIR/manifest.json'))['run_id'])")
    MANIFEST_SEED=$($PY -c "import json; print(json.load(open('$STORE_DIR/manifest.json'))['master_seed'])")
    if [ "$MANIFEST_SEED" != "$MASTER_SEED" ]; then
      echo "ERROR: RESUME with MASTER_SEED=$MASTER_SEED but store was created with $MANIFEST_SEED." >&2
      exit 1
    fi
    RESUME_ARGS="--resume"
    echo "RESUMING run $RUN_ID in $STORE_DIR"
  else
    echo "ERROR: $STORE_DIR already contains a results store." >&2
    echo "A canonical sweep must start from an empty directory (choose a new" >&2
    echo "STORE_DIR, delete the old one yourself, or pass RESUME=1 to continue" >&2
    echo "a crashed run)." >&2
    exit 1
  fi
fi

# Capture git provenance BEFORE creating STORE_DIR — otherwise the (untracked)
# output dir itself makes `git status` report the tree dirty (a false positive
# that mislabels an otherwise-clean canonical run).
GIT_SHA=$(git rev-parse --verify --quiet HEAD 2>/dev/null) || GIT_SHA=unknown
GIT_DIRTY=$(test -n "$(git status --porcelain 2>/dev/null)" && echo True || echo False)

mkdir -p "$STORE_DIR"
exec > >(tee -a "$STORE_DIR/sweep.log") 2>&1

# ── Provenance manifest (not rewritten on resume) ──
[ "$RESUME" = "1" ] && [ -f "$STORE_DIR/manifest.json" ] || $PY - <<EOF
import json, platform, sys, time
import torch
json.dump({
    'run_id': '$RUN_ID',
    'master_seed': $MASTER_SEED,
    'smoke': $SMOKE == 1,
    'date': time.strftime('%Y-%m-%dT%H:%M:%S'),
    'git_sha': '$GIT_SHA',
    'git_dirty': $GIT_DIRTY,
    'python': sys.version,
    'torch': torch.__version__,
    'platform': platform.platform(),
    'params': {
        'nws_lr_mult': $NWS_LR_MULT, 'batch_size': $BATCH_SIZE,
        'chain_steps_base': $CHAIN_STEPS_BASE,
        'sweep_ns': '$SWEEP_NS', 'sweep_ms': '$SWEEP_MS', 'sweep_Ss': '$SWEEP_SS',
        'device': '$DEVICE',
        'record_dynamics': $RECORD_DYNAMICS == 1,
        'dyn_full_groups': '$DYN_FULL_GROUPS',
        'recipe': 'AdamW lr_peak=4e-3 cosine wd=1e-2; chain l=1->l=4 warm-start '
                  '(large-noise K + near-warm-start arm); resolver: measure and target violations + '
                  'grad_clip=1.0 ema=0.999; independent store check',
    },
}, open('$STORE_DIR/manifest.json', 'w'), indent=2)
EOF

echo "============================================================"
echo "Canonical sweep"
echo "  store_dir:      $STORE_DIR"
echo "  master_seed:    $MASTER_SEED"
echo "  run_id:         $RUN_ID"
echo "  smoke:          $SMOKE"
echo "  n_gpus:         $N_GPUS   device: $DEVICE"
echo "  K (per stage):  $K"
echo "  K_nws:          $NWS_K  (lr_mult $NWS_LR_MULT)"
echo "  slice:          n=$SWEEP_NS m=$SWEEP_MS S=$SWEEP_SS"
echo "============================================================"

# Extra device args for single-process mode
CHAIN_DEV_ARGS=""
if [ "$N_GPUS" -le 1 ]; then
  CHAIN_DEV_ARGS="--device $DEVICE"
fi

# (l=1 is trained by train_sweep itself as the chain's first stage — the old
#  separate bootstrap_l1 step was absorbed; see archive/bootstrap_l1.py.)

# -------- 2. Progressive l=1 -> l=4 chain on every group with near-warm-start --------
echo
echo "[1/3] Progressive chain (l=1 -> l=4) on every (n, m, S) group, with near-warm-start arm"
$PY src/sweep/train_sweep.py --store-dir "$STORE_DIR" --n-gpus $N_GPUS \
    --K $K --batch-size $BATCH_SIZE $CHAIN_DEV_ARGS \
    --master-seed $MASTER_SEED --run-id "$RUN_ID" \
    --near-warm-start-K $NWS_K --near-warm-start-lr-mult $NWS_LR_MULT \
    --steps-base $CHAIN_STEPS_BASE $DYN_ARGS $RESUME_ARGS \
    --all-groups --sweep-ns "$SWEEP_NS" --sweep-ms "$SWEEP_MS" --sweep-Ss "$SWEEP_SS"

# -------- 2. Resolve monotonicity violations (owns its measure->resolve loop) --------
# The resolver recompiles the canonical CSV, detects CI-confirmed violations on
# all four axes, resolves them with the 5-source pool, and repeats (--max-rounds)
# until clean — exiting with compiled/sweep_results_precise.csv FRESH by
# construction. No full-grid push (quality = uniform budget knobs in the chain);
# no separate eval step (the measurement loop belongs to the resolver).
RESOLVE_ROUNDS=${RESOLVE_ROUNDS:-3}
echo
echo "[2/3] Resolve monotonicity violations (max $RESOLVE_ROUNDS rounds, K=$RESOLVE_K)"
$PY src/sweep/resolve_monotonicity_violations.py --store-dir "$STORE_DIR" \
    --n-gpus $N_GPUS --K $RESOLVE_K --batch-size $BATCH_SIZE --device "$DEVICE" \
    --grad-clip 1.0 --ema-decay 0.999 $DYN_ARGS --freeze-sources \
    --max-rounds $RESOLVE_ROUNDS --steps-base $RESOLVE_STEPS_BASE \
    --master-seed $MASTER_SEED --run-id "${RUN_ID}_resolve" \
    --near-warm-start-K $NWS_K --near-warm-start-lr-mult $NWS_LR_MULT

# -------- 5. Sanity check --------
echo
echo "[3/3] Sanity check"
$PY src/sweep/check_results.py --store-dir "$STORE_DIR" --precise --device "$DEVICE"

echo
echo "============================================================"
echo "Canonical sweep complete."
echo "Canonical results:  $STORE_DIR/compiled/sweep_results_precise.csv"
echo "Models:             $STORE_DIR/models/  ($(ls $STORE_DIR/models/ 2>/dev/null | wc -l) files)"
echo "To analyze this run, set store to $STORE_DIR in:"
echo "  results/core/nonlinear_representations/canonical_source.json"
echo "Then run:"
echo "  $PY results/core/nonlinear_representations/canonical_analysis.py"
echo "============================================================"
