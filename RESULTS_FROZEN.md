# Frozen results map

## Primary manuscript results

The primary comparison matches focal information and the learner specification, then adds only the three disjoint cross-scale histories.

- L1 rare-dormant AP: strong local 0.319844; full Mosaic 0.370776; gain 0.050932.
- L2 rare-dormant AP: strong local 0.282842; full Mosaic 0.337869; gain 0.055027.
- `resolution_observability_2023.csv`: 2023 active-month observability by resolution, used in Table 1 and Fig. 1a.
- `strong_matched_gradient_l1.csv`, `strong_matched_gradient_l2.csv`: point estimates for Fig. 1c.
- `strong_bootstrap_l1.csv`, `strong_bootstrap_l2.csv`: bin-specific 1,000-replicate paired skill-cluster intervals for Fig. 1c and Table 2 / Table S3-S4.
- `strong_context_l1.csv`, `strong_context_bootstrap_l1.csv`: L1 source attribution used in Fig. 2a and Table S6.
- `strong_quarterly_l1.csv`, `strong_quarterly_l2.csv`: 2023 block stability used in Fig. 2b and Table S2.

## Supplementary controls

- `sensitivity_exclude_highest11_primary_gbm.csv`: current strong-local 2,324-ID sensitivity at L1 and L2.
- `strong_observability_normalized_l1.csv`, `strong_observability_normalized_l2.csv`: prevalence-normalised observability sensitivity.
- `strong_linear_l1.csv`: strong-memory logistic information-set comparison.
- `neural_information_check_l1.csv`: parameter-matched neural local/cross-scale activation comparison.
- `strong_static_id_control_l1.csv`: strong local plus static skill/region/occupation identity control.

`python build_frozen_results.py` checks all compact result tables for readability and regenerates Fig. 1 and Fig. 2 in a separate `reproduced_figures/` directory.
