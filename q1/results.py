"""Shim: old `import results` / `python results.py` → full evaluation runner."""

from segpos.eval.experiments import run_full_evaluation

if __name__ == "__main__":
    run_full_evaluation()
