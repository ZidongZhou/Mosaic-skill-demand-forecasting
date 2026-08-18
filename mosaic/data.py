from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .miniparquet import ParquetFile


def _expected_months() -> list[str]:
    return [f"{year}-{month:02d}" for year in (2021, 2022, 2023) for month in range(1, 13)]


def _validate_source_table(
    data: dict[str, np.ndarray],
    id_columns: list[str],
    months: list[str],
    path: Path,
) -> None:
    """Validate the released flat Job-SDF table before reshaping.

    These checks are intentionally structural. They do not alter observations.
    """
    if months != _expected_months():
        raise ValueError(
            f"Unexpected month axis in {path.name}: expected 2021-01..2023-12"
        )

    n_rows = len(data[id_columns[0]])
    if any(len(data[c]) != n_rows for c in id_columns + months):
        raise ValueError(f"Column-length mismatch in {path.name}")

    # The released ID axes are zero-based contiguous integer identifiers.
    cardinalities: list[int] = []
    for col in id_columns:
        ids = np.asarray(data[col])
        unique = np.unique(ids)
        expected = np.arange(len(unique), dtype=unique.dtype)
        if not np.array_equal(unique, expected):
            raise ValueError(f"Non-contiguous or non-zero-based identifiers in {path.name}:{col}")
        cardinalities.append(len(unique))

    expected_rows = int(np.prod(cardinalities, dtype=np.int64))
    if n_rows != expected_rows:
        raise ValueError(
            f"Incomplete Cartesian product in {path.name}: {n_rows} rows, expected {expected_rows}"
        )

    keys = np.column_stack([np.asarray(data[c]) for c in id_columns])
    if len(np.unique(keys, axis=0)) != n_rows:
        raise ValueError(f"Duplicate identifier combinations in {path.name}")

    values = np.column_stack([np.asarray(data[m], dtype=float) for m in months])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite demand value in {path.name}")
    if (values < 0).any():
        raise ValueError(f"Negative demand value in {path.name}")


def _read_table(path: Path, id_columns: list[str]) -> tuple[dict[str, np.ndarray], list[str]]:
    """Read and structurally validate one released Job-SDF Parquet table.

    The bundled reader is deliberately narrow: it supports the flat, Snappy-compressed,
    dictionary/RLE encoded files distributed with Job-SDF. If pyarrow is available,
    users may replace this function with pandas.read_parquet without changing any
    downstream code.
    """
    parquet = ParquetFile(path)
    months = sorted(c for c in parquet.column_names if len(c) == 7 and c[4] == "-")
    data = parquet.read(id_columns + months)
    _validate_source_table(data, id_columns, months, path)
    # Job-SDF rows are ordered into a dense Cartesian product before reshaping.
    keys = tuple(data[c] for c in reversed(id_columns))
    order = np.lexsort(keys)
    return {k: v[order] for k, v in data.items()}, months


@dataclass
class DemandLattice:
    """A cross-classified skill-demand lattice.

    Shapes
    ------
    market:      [skills, time]
    region:      [skills, regions, time]
    occupation:  [skills, occupations, time]
    bottom:      [skills, regions, occupations, time]
    """

    market: np.ndarray
    region: np.ndarray
    occupation: np.ndarray
    bottom: np.ndarray
    months: list[str]
    occupation_level: str

    @property
    def n_skills(self) -> int:
        return int(self.market.shape[0])

    @property
    def n_regions(self) -> int:
        return int(self.region.shape[1])

    @property
    def n_occupations(self) -> int:
        return int(self.occupation.shape[1])

    @property
    def n_periods(self) -> int:
        return int(self.market.shape[-1])

    @classmethod
    def from_job_sdf(
        cls, data_root: str | Path, occupation_level: str = "r1"
    ) -> "DemandLattice":
        if occupation_level not in {"r1", "r2"}:
            raise ValueError("occupation_level must be 'r1' or 'r2'")

        demand = Path(data_root) / "demand"
        market, months = _read_table(demand / "r0.parquet", ["skill_id", "r0_id"])
        region, mr = _read_table(demand / "region.parquet", ["skill_id", "region_id"])
        occupation, mo = _read_table(
            demand / f"{occupation_level}.parquet",
            ["skill_id", f"{occupation_level}_id"],
        )
        bottom, mb = _read_table(
            demand / f"{occupation_level}-region.parquet",
            ["skill_id", "region_id", f"{occupation_level}_id"],
        )
        if not (months == mr == mo == mb):
            raise ValueError("Month columns differ across Job-SDF tables")

        s = len(np.unique(market["skill_id"]))
        r = len(np.unique(region["region_id"]))
        o = len(np.unique(occupation[f"{occupation_level}_id"]))
        t = len(months)

        def values(table: dict[str, np.ndarray]) -> np.ndarray:
            return np.stack([table[m] for m in months], axis=-1).astype(np.float32)

        lattice = cls(
            market=values(market).reshape(s, t),
            region=values(region).reshape(s, r, t),
            occupation=values(occupation).reshape(s, o, t),
            bottom=values(bottom).reshape(s, r, o, t),
            months=months,
            occupation_level=occupation_level,
        )
        lattice.validate()
        return lattice

    @classmethod
    def from_npz(cls, path: str | Path) -> "DemandLattice":
        z = np.load(path, allow_pickle=True)
        level_key = "occupation_level" if "occupation_level" in z.files else "level"
        lattice = cls(
            market=z["market"].astype(np.float32),
            region=z["region"].astype(np.float32),
            occupation=z["occupation"].astype(np.float32),
            bottom=z["bottom"].astype(np.float32),
            months=[str(x) for x in z["months"]],
            occupation_level=str(z[level_key].item()),
        )
        lattice.validate()
        return lattice

    def to_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            market=self.market,
            region=self.region,
            occupation=self.occupation,
            bottom=self.bottom,
            months=np.asarray(self.months),
            occupation_level=np.asarray(self.occupation_level),
        )

    def validate(self) -> None:
        arrays = {
            "market": self.market,
            "region": self.region,
            "occupation": self.occupation,
            "bottom": self.bottom,
        }
        for name, arr in arrays.items():
            if not np.isfinite(arr).all():
                raise ValueError(f"Non-finite values in {name} lattice")
            if (arr < 0).any():
                raise ValueError(f"Negative values in {name} lattice")

        errors = {
            "bottom_to_market": np.max(
                np.abs(self.bottom.sum((1, 2)) - self.market)
            ),
            "bottom_to_region": np.max(np.abs(self.bottom.sum(2) - self.region)),
            "bottom_to_occupation": np.max(
                np.abs(self.bottom.sum(1) - self.occupation)
            ),
        }
        if max(errors.values()) != 0:
            raise ValueError(f"Incoherent demand lattice: {errors}")


class SkillWindowDataset(Dataset):
    """Rolling windows indexed by skill and forecast origin."""

    def __init__(
        self,
        lattice: DemandLattice,
        lookback: int,
        horizon: int,
        target_starts: Iterable[int],
    ) -> None:
        self.lattice = lattice
        self.lookback = int(lookback)
        self.horizon = int(horizon)
        self.target_starts = [int(t) for t in target_starts]
        self.samples = [
            (skill, target_start)
            for target_start in self.target_starts
            for skill in range(lattice.n_skills)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        skill, target_start = self.samples[index]
        begin = target_start - self.lookback
        end = target_start + self.horizon
        lattice = self.lattice
        return {
            "skill_index": torch.tensor(skill),
            "target_start": torch.tensor(target_start),
            "market_history": torch.from_numpy(
                lattice.market[skill, begin:target_start]
            ),
            "region_history": torch.from_numpy(
                lattice.region[skill, :, begin:target_start]
            ),
            "occupation_history": torch.from_numpy(
                lattice.occupation[skill, :, begin:target_start]
            ),
            "bottom_history": torch.from_numpy(
                lattice.bottom[skill, :, :, begin:target_start]
            ),
            "bottom_target": torch.from_numpy(
                lattice.bottom[skill, :, :, target_start:end]
            ),
        }
