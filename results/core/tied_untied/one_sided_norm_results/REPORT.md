# One-sided unit-norm follow-up

Five input features, two hidden dimensions, sparsity 0.9; output ReLU and learned output bias.
MSE averages over examples and all five features.

## Previous experiments

The original notebook started from one trained tied solution, ran encoder-only normalization twice
(once for its table and once for its plot), and ran one encoder-normalized warm start from a free solution.
The displayed table reported about +1% improvement for the tied start and -32% for the free warm start.
It also tested both-sided normalization and free models with/without weight decay.
These were not a multi-start search; losses used separate fresh evaluation draws.

## New search

- `explore_free_cartesian`: 12 runs, 30,000 steps, batch 2,048, learning rate 0.003.
- `explore_tied_cartesian`: 12 runs, 30,000 steps, batch 2,048, learning rate 0.003.
- `explore_unit_decoder_angle`: 36 runs, 30,000 steps, batch 2,048, learning rate 0.003.
- `explore_unit_decoder_cartesian`: 12 runs, 30,000 steps, batch 2,048, learning rate 0.003.
- `explore_unit_encoder_angle`: 36 runs, 30,000 steps, batch 2,048, learning rate 0.003.
- `explore_unit_encoder_cartesian`: 12 runs, 30,000 steps, batch 2,048, learning rate 0.003.
- `refine_free`: 5 runs, 60,000 steps, batch 4,096, learning rate 0.001.
- `refine_tied`: 5 runs, 60,000 steps, batch 4,096, learning rate 0.001.
- `refine_unit_decoder`: 8 runs, 60,000 steps, batch 4,096, learning rate 0.001.
- `refine_unit_encoder`: 8 runs, 60,000 steps, batch 4,096, learning rate 0.001.

One-sided starts cover tied warm starts, random directions with read/write lengths 1.05–8,
independent random reads/writes, perturbed pentagons, and free-model warm starts with and without
partner rescaling to preserve diagonal responses (off-diagonal responses still change).
Both angle parameters and forward-normalized Cartesian parameters are tested.
Adam uses a constant learning rate for half the run and cosine cooldown to 1% thereafter.
All models within a group receive the same fresh training batches; different groups use different streams.
Refinement includes the four lowest-validation-loss candidates and the best candidate from every initial family.
Source tied/free warm starts have 30,000/60,000 preceding training steps; this is an optimization search, not a comparison of training efficiency.

## Held-out results

Checkpoint and model selection use 131,072 validation examples; these results use a separate shared draw of 1,000,000 examples.
Improvement is relative to the validation-selected tied control, evaluated on those same test examples.

| Constraint | Test MSE | Improvement vs. tied | Paired 95% interval |
| --- | ---: | ---: | ---: |
| Tied | 5.237715e-03 | +0.00% | reference |
| Untied, unconstrained | 3.113272e-03 | +40.56% | [+40.11%, +41.01%] |
| Unit writes (SAE decoder analogue) | 5.237775e-03 | -0.0012% | [-0.0025%, +0.0002%] |
| Unit reads | 5.237587e-03 | +0.0024% | [-0.0001%, +0.0050%] |

Intervals quantify test-sampling error, not variation across training procedures or uncertainty about the global optimum.
No global-optimality claim follows from these numerical searches.

The SAE analogy concerns which vectors are normalized: toy encoder/write columns correspond to SAE decoder/dictionary columns.
The objective here is toy feature-reconstruction MSE, not SAE activation-reconstruction loss plus an activation sparsity penalty.

## Geometry of selected solutions

| Constraint | Mean read/write angle | Free-side norm range | Mean self-response | Asymmetry |
| --- | ---: | ---: | ---: | ---: |
| Unit writes (SAE decoder analogue) | 0.03° | 1.210–1.211 | 1.210 | 0.0006 |
| Unit reads | 0.07° | 1.208–1.214 | 1.211 | 0.0016 |

## Interpretation

These experiments test whether other initial configurations or parameterizations recover a one-sided advantage.
The best one-sided solutions return to nearly aligned reads and writes; the larger free-side norms also accommodate self-responses above one and negative output biases.
For example, unit writes with aligned reads of length 1.21 implement the same reconstruction operator as a tied model with all vectors of length $\sqrt{1.21}$.
Thus a free-side norm above one need not imply a directional benefit from untying.

The poor free-model warm start in the original notebook was not evidence that optimization effects had been ruled out:
normalizing a side changes the function, and the new warm starts that preserve diagonal responses can recover the tied-level solution.
Preserving diagonal responses does not preserve the entire function or remove all optimization difficulties.

One-sided normalization does not mathematically require alignment, and a broader search could still find better solutions.
Even with both sides at unit norm, alignment follows only if the own-feature response is exactly one; unit norms alone do not force symmetry.


## Files and reproduction

- [Full result and selected weights](summary.json).
- `explore_*.json` and `refine_*.json`: every run, validation trace, selected checkpoint weights, and exact training settings.
- [Search plot](search_results.png): all exploratory validation results; held-out test results are in the table above.
- [Experiment script](../one_sided_norm_experiments.py).

```sh
python results/core/tied_untied/one_sided_norm_experiments.py --phase explore
python results/core/tied_untied/one_sided_norm_experiments.py --phase refine --steps 60000 --batch 4096
MPLCONFIGDIR=/tmp/one-sided-norm-mpl python results/core/tied_untied/one_sided_norm_experiments.py --phase report
```
