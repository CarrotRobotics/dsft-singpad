# Dual-State Familiarity Tracing (DSFT) for sight-singing

Code for the paper *Dual-State Familiarity Tracing: Interpretable Knowledge Tracing for Sight-Singing Practice* (Yan Li and Xu Wang, AIES 2026).

DSFT is an interpretable knowledge-tracing model for sung exercises. It combines

- a **long-term layer**: a three-level hidden Markov model of each learner's familiarity with each musical concept
  (unfamiliar, partially familiar, familiar), and
- a **short-term layer**: a hidden Markov chain over the state of the current take (from on track to lost),
  together with a difficulty offset for each note and learner-ability features computed from earlier takes.

On the public [SingPAD](https://github.com/itec-hust/singKT-dataset) dataset (1,081,876 note attempts, 1,074 learners)
it reaches an AUC of 0.847 in five-fold learner-level cross-validation, on par with a tuned DKT and above SAKT, AKT,
simpleKT and a strong logistic-regression baseline, with about 5,000 parameters.

## Setup

Python 3.10 or later. Everything runs on a CPU; no GPU is needed.

```bash
pip install -r requirements.txt
git clone https://github.com/itec-hust/singKT-dataset data/singKT-dataset
```

## Reproducing the paper

Run the steps in order from the repository root. Each step writes to `outputs/` (change with `--out`).

| Step | Command | Produces | Time on a 2-core CPU |
| --- | --- | --- | --- |
| 1 | `python scripts/01_prepare_data.py` | `data/singpad.npz`, one row per sung note | 2 min |
| 2 | `python scripts/02_familiarity_models.py` | BKT, BKT-Forget and the familiarity layer, per fold | 20 min |
| 3 | `python scripts/03_dsft.py` | choice of take states, DSFT core and final model, all ablations | 3–4 h |
| 4 | `python scripts/04_baselines.py` | PFA, logistic regression, DKT, SAKT, AKT, simpleKT | 2 h |
| 5 | `python scripts/05_results.py` | Table 1, Table 2, AUC by practice gap, bootstrap intervals (`results/tables.md`) | 30 min |
| 6 | `python scripts/06_analyses.py` | error runs, forgetting after gaps, the three readable take states (`results/analyses.md`) | 2 min |
| 7 | `python scripts/07_figures.py` | Figs. 1–3 as PDF and PNG (`figures/`) | 1 min |
| 8 | `python scripts/08_small_data.py` | optional: training on 17 and 43 learners (Section 5.3) | 1–2 h |
| 9 | `python scripts/09_seeds.py` | optional: DSFT, DKT and simpleKT with two further seeds (Section 5.1, `results/seeds.md`) | 1–1.5 h |

`bash run_all.sh` runs steps 1–7. Add `--quick` to any step (or `bash run_all.sh --quick`) to check that the
pipeline runs end to end on 150 learners with very short training; its numbers are not meaningful.

`04_baselines.py` uses the DKT and SAKT settings chosen in the paper (DKT: 100 hidden units, learning rate 3e-3;
SAKT: 128-dimensional embeddings, learning rate 3e-3). Pass `--tune` to rerun that search on fold 0's validation learners.

### Expected results (Table 1)

| Model | Parameters | AUC (five-fold mean ± SD) | RMSE |
| --- | --- | --- | --- |
| BKT | 1,080 | 0.6835 ± 0.0087 | 0.413 |
| PFA | 810 | 0.6927 ± 0.0085 | 0.412 |
| BKT-Forget | 1,350 | 0.7097 ± 0.0055 | 0.408 |
| SAKT (tuned) | 556,417 | 0.8281 ± 0.0061 | 0.366 |
| AKT | 147,744 | 0.8375 ± 0.0045 | 0.360 |
| Logistic regression | 2,867 | 0.8376 ± 0.0039 | 0.361 |
| simpleKT | 147,740 | 0.8424 ± 0.0045 | 0.358 |
| DKT (tuned) | 127,894 | 0.8466 ± 0.0042 | 0.355 |
| **DSFT** | **5,062** | **0.8472 ± 0.0037** | 0.356 |

Folds, validation splits and seeds are fixed in the code, so a full run should reproduce these numbers closely.
Small differences (around ±0.001 AUC) can come from library versions and floating-point order, mainly for the deep models.
The optional small-data step trains DKT with early stopping, so its DKT numbers can differ slightly from Section 5.3,
which reports that DKT is sensitive to training settings with so few learners.

### Seed robustness (step 9)

| Model | Seed 0 | Seed 1 | Seed 2 |
| --- | --- | --- | --- |
| DSFT | 0.8472 | 0.8470 | 0.8468 |
| DKT (tuned) | 0.8466 | 0.8467 | 0.8469 |
| simpleKT | 0.8424 | 0.8424 | 0.8426 |

DSFT and DKT differ by less than 0.001 AUC for every seed; DSFT stays 0.004 to 0.005 AUC above simpleKT for every seed.

## Code layout

```
dsft/
  data.py           SingPAD loading, note-level arrays, learner folds and validation splits
  familiarity.py    long-term layer and the BKT-family baselines (per-concept HMMs)
  take_layer.py     short-term take layer: C take states, note difficulty, learner ability
  features.py       PFA counts, in-take features, learner-ability features, practice gaps
  classic_kt.py     PFA and DKT
  attention_kt.py   SAKT, AKT and simpleKT (re-implemented, one attention block)
  evaluation.py     per-fold AUC, RMSE, learner-level bootstrap
scripts/            the steps above
```

All models are written in JAX. Every prediction for a note uses only the same learner's earlier notes, and the
learner-ability features use only earlier takes, never the current one.

## Data

SingPAD is released by its authors at <https://github.com/itec-hust/singKT-dataset>; this repository does not
redistribute it. Please cite their paper when using the data:

> Y. Zhang, Y. Zhang, W. Xu, Z. Wang, and J. Sun. SingPAD: A knowledge tracing dataset based on music performance
> assessment. In *Proceedings of the 17th International Conference on Educational Data Mining (EDM)*, 2024.

## Citation

If you use this code, please cite our paper:

> Y. Li and X. Wang. Dual-State Familiarity Tracing: Interpretable Knowledge Tracing for Sight-Singing Practice.
> 2026 International Conference on AI and Education Systems (AIES 2026). Full citation to be added after publication.

## License

MIT (see `LICENSE`).
