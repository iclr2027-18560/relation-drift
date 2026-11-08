# Learning Time-Varying Relational Constraints

Code for the synthetic sheaf model, the pair estimators, and the dataset checks used in the paper.

Run every command from this directory. Paths such as `data/` and `results/` are relative to the repository root.

## Layout

| Path | Role |
|---|---|
| `src/run/` | Cayley sheaf (`temporal_sheaf_v35.py`), trainer (`train_v35.py`), synthetic and Elliptic runners |
| `src/run_sheaf_benchmark.py` | Synthetic graph generator used by the benchmarks |
| `src/*.py` | Loaders for Elliptic, IBM HI-Small, MathOverflow, protein, and the other injects |
| `experiments/` | Table 1, baselines, jump checks, and dataset scripts |
| `data/` | Datasets already in the tree |
| `results/` | JSON written by the scripts |

## Setup

Python 3.10. Install PyTorch, then PyTorch Geometric, then the rest:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch
pip install torch-geometric
pip install -r requirements.txt
```

`requirements.txt` does not install PyTorch. On a GPU machine, install the CUDA build of PyTorch from the PyTorch site before `torch-geometric`.

## Paper runs

Synthetic Table 1 (seeds 42–46, clean noise, four drifts). This is a long run.

```bash
python experiments/run_table1_multiseed.py
```

Smoke test of one cell:

```bash
python experiments/run_table1_multiseed.py --quick --seeds 42 --drifts single_jump --models mlp dual
```

Output: `results/table1_multiseed.json`.

Leak-free V3.5 synthetic benchmark (the single-seed numbers). `--no-jump-labels` keeps training on `t < 0.5` with no planted jump labels:

```bash
python src/run/run_v35_benchmark.py --no-jump-labels --noise clean
```

A shorter run:

```bash
python src/run/run_v35_benchmark.py --quick --no-jump-labels --noise clean --drift single_jump
```

Output: `results/week8_v35_unsupervised_jump.json`.

Five-seed jump comparison of V3.5 and DualPath:

```bash
python experiments/run_jump_multiseed.py
```

Output: `results/jump_multiseed_unsupervised.json`.

Elliptic, train timesteps 1–34 and test timesteps 35–49. The three CSVs are already in `data/elliptic/`.

```bash
python src/run/run_elliptic_benchmark.py --data_dir data/elliptic
python experiments/run_elliptic_dynamic_baselines.py --models dual evolvegcn tgn
```

Outputs: `results/elliptic_v35_results.json` and `results/elliptic_dynamic_baselines.json`. Add `--quick` for 40 epochs on the baseline script.

Separate baseline scripts, if you want one model at a time:

```bash
python experiments/run_tgn_baseline.py
python experiments/run_evolvegcn_baseline.py
python experiments/run_dualpath_gnn_baseline.py
```

## Other checks

These read the files under `data/` and write JSON under `results/`.

```bash
python experiments/run_kaggle_inject.py          # data/kaggle/HI-Small_Trans.csv
python experiments/run_mathoverflow_inject.py    # data/mathoverflow/
python experiments/run_elliptic_inject.py        # data/elliptic/
python experiments/run_pharma_inject.py          # data/pdb/
python experiments/run_procrustes_baseline.py
python experiments/run_gld_residual_v35.py
```

Each script prints its own `--help`.
