# Data requirements

The analysis uses the public processed Job-SDF release from https://github.com/Job-SDF/benchmark. The Job-SDF source files are obtained from the official benchmark repository and are not redistributed in this archive.

`prepare_lattice.py` expects the repository's `dataset/` directory, containing `demand/` with:

- `r0.parquet`
- `region.parquet`
- `r1.parquet` and `r1-region.parquet` for L1
- `r2.parquet` and `r2-region.parquet` for L2

The processed files used for this revision contain 36 monthly columns from 2021-01 to 2023-12. Direct summation of the bottom lattice is checked against market, region and occupation margins before model fitting.

The released processed files used here contain **2,335 contiguous numeric skill indices (0-2334)**, while the Job-SDF main text reports 2,324 standardised skills and Appendix F.2 reports 2,335 skills over 36 months. Repository state is recorded in `SOURCE_REPOSITORY_SNAPSHOT.txt`, and exact analysed files are fixed by size and SHA-256 digest in `SOURCE_DATA_MANIFEST.csv`.

The skill-count sensitivity excludes the 11 highest-numbered IDs (2324-2334), retaining 2,324 IDs, and reruns the same strong-local and full-Mosaic specification used in the primary analysis at both L1 and L2 (`frozen_results/sensitivity_exclude_highest11_primary_gbm.csv`). It is a deterministic size sensitivity, not a reconstruction of a distinct source taxonomy.

## Parquet reader dependency

The bundled `mosaic/miniparquet.py` reader supports the flat, Snappy-compressed numeric Job-SDF files used here. It requires Apache Thrift and the system Snappy runtime (`libsnappy.so.1`). PyArrow is not required by the released analysis code.
