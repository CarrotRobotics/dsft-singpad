"""Shared command-line options and paths for the experiment scripts."""
import argparse, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def args(extra=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/singpad.npz", help="output of 01_prepare_data.py")
    p.add_argument("--out", default="outputs", help="directory for predictions, results and figures")
    p.add_argument("--quick", action="store_true", help="150 learners and very short training: checks the pipeline runs, numbers are meaningless")
    if extra:
        extra(p)
    a = p.parse_args()
    for d in ("preds", "results", "figures"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    return a


def path(a, *parts):
    return os.path.join(a.out, *parts)


def log(*m):
    print(time.strftime("%H:%M:%S"), *m, flush=True)


# training lengths: full runs reproduce the paper; --quick only checks that everything runs
def iters(a, full):
    return max(2, full // 50) if a.quick else full
