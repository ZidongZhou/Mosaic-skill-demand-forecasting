# Supplementary code metadata

Article title: The resolution paradox in skill-demand forecasting: Mosaic cross-scale learning across regional and occupational labour markets  
Journal: Applied Intelligence  
Authors: Zidong Zhou; Yue Meng; Xuanhe Wang; Ke Liu  
Affiliations: 1 School of Digital Arts, Jiangsu Vocational Institute of Commerce, Nanjing 211168, China; 2 School of Economics and Management, Southeast University, Nanjing 211189, China; 3 School of Computer Science and Engineering, Sun Yat-sen University, Guangzhou 510006, China; 4 School of Physical Education and Sports, Jining Normal University, Ulanqab 012000, China  
Corresponding author: Yue Meng  
Corresponding author e-mail: 230268426@seu.edu.cn

# Mosaic cross-scale skill-demand forecasting

This archive accompanies the Applied Intelligence manuscript **“The resolution paradox in skill-demand forecasting: Mosaic cross-scale learning across regional and occupational labour markets.”** The primary analysis treats Mosaic as an information-design framework. Strong local and full Mosaic use the same LightGBM specification and the same focal-cell information; full Mosaic additionally receives the three disjoint cross-scale histories.

## Installation

Minimum dependencies are listed in `requirements.txt`; exact tested versions are recorded in `requirements-tested.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

The bundled narrow Job-SDF Parquet reader requires Apache Thrift and the system Snappy runtime (`libsnappy.so.1`).

## Data preparation

Clone the public Job-SDF repository and check out the provenance snapshot used by the analysis:

```bash
git clone https://github.com/Job-SDF/benchmark.git
cd benchmark
git checkout 00583fc0b0c70b4ed8df6d579e760e89d57ddbb1
```

The snapshot was verified on 8 August 2026. Exact analysed files are fixed independently by byte size and SHA-256 digest in `SOURCE_DATA_MANIFEST.csv`. `prepare_lattice.py` expects the checked-out repository's `dataset/` directory:

```bash
python prepare_lattice.py --dataset /path/to/benchmark/dataset --level r1 --output r1.npz
python prepare_lattice.py --dataset /path/to/benchmark/dataset --level r2 --output r2.npz
```

The generated lattices contain 2,335 skill IDs, 7 regions, 14 L1 or 52 L2 occupations, and 36 monthly observations from January 2021 through December 2023.

## Forecasting protocol

The primary comparison uses:

- 12 recent focal months;
- four longer-memory focal summaries: cumulative pre-origin active months, the corresponding active-month ratio, months since the latest positive, and a pre-window-positive indicator;
- a three-month activation horizon;
- three 2022 training target blocks (January-March, April-June and July-September);
- October-December 2022 excluded from primary GBM fitting and used as the validation block for the secondary neural comparison;
- four 2023 evaluation blocks. The primary GBM models are fitted once on the three 2022 training blocks and then held fixed; only predictor histories advance as additional months become observed.

A rare-dormant cell has at most two active months in the preceding 12 months and zero demand in the final pre-origin month. The event is at least one positive demand month in the following three months.

## Disjoint Mosaic contexts

For focal demand `Y(s,r,o,t)`:

- regional context: `Y^R(s,r,t) - Y(s,r,o,t)`;
- occupational context: `Y^O(s,o,t) - Y(s,r,o,t)`;
- residual-market context: `Y^M(s,t) - Y^R(s,r,t) - Y^O(s,o,t) + Y(s,r,o,t)`.

These histories are mutually non-overlapping by construction. The test suite checks the lattice arithmetic and context decomposition.

## Primary matched experiment

```bash
python strong_matched_cross_scale.py --lattice r1.npz --level L1 --output outputs/strong --save-scores
python strong_matched_cross_scale.py --lattice r2.npz --level L2 --output outputs/strong --save-scores
```

The script writes rare-dormant summaries, the observability gradient, quarterly results and the prevalence-normalised gradient. The primary GBM pipeline uses the fixed LightGBM specification coded in `strong_matched_cross_scale.py`; it does not perform a separate validation-based hyperparameter search.

The primary paired skill-cluster intervals resample only the skill clusters represented in each evaluated subset. The implementation sorts scores once and computes exact weighted AP with tie handling.

```bash
python strong_cluster_bootstrap.py \
  --scores outputs/strong/strong_matched_scores_L1.npz --level L1 \
  --output outputs/strong/strong_bootstrap_L1.csv --replicates 1000

python strong_cluster_bootstrap.py \
  --scores outputs/strong/strong_matched_scores_L2.npz --level L2 \
  --output outputs/strong/strong_bootstrap_L2.csv --replicates 1000
```

## Resolution and source attribution

Table 1 / Fig. 1a observability statistics are generated directly from the four 2023 forecast origins:

```bash
python resolution_observability.py --l1 r1.npz --l2 r2.npz \
  --out outputs/resolution_observability_2023.csv
```

The L1 source analysis fits every single-context and leave-one-context-out information set:

```bash
python strong_context_ablation.py --lattice r1.npz \
  --output outputs/strong_context_l1.csv \
  --scores outputs/strong_context_l1_scores.npz

python strong_context_bootstrap.py --scores outputs/strong_context_l1_scores.npz \
  --output outputs/strong_context_bootstrap_l1.csv --replicates 500
```

## Additional controls used in Online Resource 1

The 2,324-ID sensitivity uses the current strong-local primary specification at both resolutions:

```bash
python primary_gbm_sensitivity.py --l1 r1.npz --l2 r2.npz --skills 2324 \
  --output outputs/sensitivity_exclude_highest11_primary_gbm.csv
```

The strong-memory logistic comparison and the static-identity control are reproduced with:

```bash
python strong_linear_cross_scale.py --lattice r1.npz \
  --out outputs/strong_linear_l1.csv

python strong_static_id_control.py --lattice r1.npz \
  --out outputs/strong_static_id_control_l1.csv
```

The supplementary neural comparison uses the capacity-matched `local_matched` and `mosaic` variants in `neural_information_check.py`, seed 11 and the 2022 validation-best checkpoint. The script computes rare-dormant AP from its test predictions and writes `neural_information_check_l1.csv`; the frozen copy contains the two rows used in the manuscript supplement.

```bash
python neural_information_check.py --lattice r1.npz --output outputs/neural_check --seed 11 --epochs 4
```

## Frozen results and figures

`frozen_results/` contains only compact result tables cited by the manuscript or Online Resource 1. Run:

```bash
python build_frozen_results.py
```

This checks that the compact tables are present/readable and regenerates `Fig1` and `Fig2` into `reproduced_figures/`. Regenerated vector files are written separately and do not overwrite the archived figures.

## Reproducibility checks

```bash
python -m pytest -q
sha256sum -c SHA256SUMS.txt
```

`SHA256SUMS.txt` verifies the pristine distributed archive. The Job-SDF source files are obtained from the official benchmark repository and are not redistributed here. Large cell-level prediction files and model checkpoints are not required for the compact frozen-result checks and are not included.
