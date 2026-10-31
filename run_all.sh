#!/usr/bin/env bash
# Reproduce the paper (steps 1-7). Pass --quick for a fast end-to-end check on 150 learners.
set -euo pipefail
cd "$(dirname "$0")"
if [ ! -d data/singKT-dataset ]; then
  git clone https://github.com/itec-hust/singKT-dataset data/singKT-dataset
fi
python scripts/01_prepare_data.py --raw data/singKT-dataset --data data/singpad.npz
for step in 02_familiarity_models 03_dsft 04_baselines 05_results 06_analyses 07_figures; do
  echo "=== $step"
  python "scripts/$step.py" --data data/singpad.npz --out outputs "$@"
done
echo "Tables: outputs/results/tables.md   Analyses: outputs/results/analyses.md   Figures: outputs/figures/"
