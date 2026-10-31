"""Step 1. Convert the raw SingPAD tables into note-level arrays.

    git clone https://github.com/itec-hust/singKT-dataset data/singKT-dataset
    python scripts/01_prepare_data.py --raw data/singKT-dataset --data data/singpad.npz
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from dsft import data

p = argparse.ArgumentParser()
p.add_argument("--raw", default="data/singKT-dataset")
p.add_argument("--data", default="data/singpad.npz")
a = p.parse_args()
os.makedirs(os.path.dirname(a.data) or ".", exist_ok=True)
n = data.prepare(a.raw, a.data)
print(f"saved {n:,} note attempts to {a.data}")
