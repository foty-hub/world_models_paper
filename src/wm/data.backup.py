from __future__ import annotations

import bisect
import operator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any

import grain.python as grain
import numpy as np


@dataclass(frozen=True)
class NumpyRun:
    path: Path
    obs_path: Path
    length: int
    obs_shape: tuple[int, ...]
    obs_dtype: np.dtype


@dataclass(frozen=True)
class NumpyRunCatalog:
    runs: tuple[NumpyRun, ...]
    offsets: tuple[int, ...]
    root: Path | None = None

    @classmethod
    def from_root(cls, root: str | PathLike[str] = "data") -> NumpyRunCatalog:
        return cls.from_run_dirs(discover_run_dirs(root), root=Path(root))

    @classmethod
    def from_run_dirs(
        cls,
        run_dirs: list[Path] | tuple[Path, ...],
        *,
        root: Path | None = None,
    ) -> NumpyRunCatalog:
        runs: list[NumpyRun] = []
        expected_shape: tuple[int, ...] | None = None
        expected_dtype: np.dtype | None = None

        for run_dir in run_dirs:
            obs_path = run_dir / "obs.npy"
            if not obs_path.is_file():
                raise FileNotFoundError(f"Missing observation file: {obs_path}")

            obs = np.load(obs_path, mmap_mode="r")
            if obs.ndim < 1:
                raise ValueError(f"{obs_path} must have a leading frame dimension")
            if obs.shape[0] == 0:
                raise ValueError(f"{obs_path} contains no observations")

            obs_shape = tuple(int(dim) for dim in obs.shape[1:])
            obs_dtype = np.dtype(obs.dtype)
            length = int(obs.shape[0])
            del obs

            if expected_shape is None:
                expected_shape = obs_shape
                expected_dtype = obs_dtype
            elif obs_shape != expected_shape or obs_dtype != expected_dtype:
                raise ValueError(
                    f"{obs_path} has observations with shape {obs_shape} and "
                    f"dtype {obs_dtype}; expected shape {expected_shape} and "
                    f"dtype {expected_dtype}"
                )

            runs.append(
                NumpyRun(
                    path=run_dir,
                    obs_path=obs_path,
                    length=length,
                    obs_shape=obs_shape,
                    obs_dtype=obs_dtype,
                )
            )

        if not runs:
            raise ValueError("No complete NumPy runs found")

        offsets = [0]
        for run in runs:
            offsets.append(offsets[-1] + run.length)

        return cls(runs=tuple(runs), offsets=tuple(offsets), root=root)

    @property
    def num_runs(self) -> int:
        return len(self.runs)

    @property
    def obs_shape(self) -> tuple[int, ...]:
        return self.runs[0].obs_shape

    @property
    def obs_dtype(self) -> np.dtype:
        return self.runs[0].obs_dtype

    @property
    def total_observations(self) -> int:
        return self.offsets[-1]

    def lookup(self, index: int) -> tuple[int, int]:
        index = operator.index(index)
        if index < 0 or index >= self.total_observations:
            raise IndexError(
                f"Observation index {index} out of range for "
                f"{self.total_observations} observations"
            )

        run_index = bisect.bisect_right(self.offsets, index) - 1
        local_index = index - self.offsets[run_index]
        return run_index, local_index


class ObservationDataSource:
    """Random-access PyGrain data source for individual VAE observations."""

    def __init__(
        self,
        root: str | PathLike[str] = "data",
        *,
        normalize: bool = True,
        catalog: NumpyRunCatalog | None = None,
        data_frac: float | None = None,
        seed: int = 0,
    ):
        self.catalog = catalog if catalog is not None else NumpyRunCatalog.from_root(root)
        self.normalize = normalize
        self._indices = _fractional_subset_indices(
            self.catalog.total_observations,
            data_frac=data_frac,
            seed=seed,
        )
        self._obs_arrays: list[np.memmap] | None = None

    def __len__(self) -> int:
        if self._indices is not None:
            return len(self._indices)
        return self.catalog.total_observations

    def __getitem__(self, index: int) -> np.ndarray:
        index = operator.index(index)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Observation index {index} out of range for "
                f"{len(self)} observations"
            )
        if self._indices is not None:
            index = int(self._indices[index])
        run_index, local_index = self.catalog.lookup(index)
        obs = self._load_obs_arrays()[run_index][local_index]
        if self.normalize:
            return obs.astype(np.float32) / 255.0
        return np.asarray(obs)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_obs_arrays"] = None
        return state

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(num_runs={self.catalog.num_runs}, "
            f"num_records={len(self)}, "
            f"total_records={self.catalog.total_observations}, "
            f"normalize={self.normalize})"
        )

    def _load_obs_arrays(self) -> list[np.memmap]:
        if self._obs_arrays is None:
            self._obs_arrays = [
                np.load(run.obs_path, mmap_mode="r") for run in self.catalog.runs
            ]
        return self._obs_arrays


def discover_run_dirs(root: str | PathLike[str] = "data") -> list[Path]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Data root does not exist: {root_path}")

    return [
        run_dir
        for run_dir in sorted(root_path.glob("run_*"))
        if run_dir.is_dir() and _run_is_complete(run_dir)
    ]


def make_observation_loader(
    root: str | PathLike[str] = "data",
    *,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 0,
    num_epochs: int | None = None,
    drop_remainder: bool = True,
    normalize: bool = True,
    data_frac: float | None = None,
    worker_count: int | None = 0,
    worker_buffer_size: int = 1,
) -> grain.DataLoader:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if worker_count is None or worker_count > 0:
        _ensure_grain_worker_flags_accessible()

    data_source = ObservationDataSource(
        root,
        normalize=normalize,
        data_frac=data_frac,
        seed=seed,
    )
    sampler = grain.IndexSampler(
        num_records=len(data_source),
        shuffle=shuffle,
        seed=seed,
        num_epochs=num_epochs,
    )
    return grain.DataLoader(
        data_source=data_source,
        sampler=sampler,
        operations=[grain.Batch(batch_size, drop_remainder=drop_remainder)],
        worker_count=worker_count,
        worker_buffer_size=worker_buffer_size,
    )


def _fractional_subset_indices(
    num_records: int,
    *,
    data_frac: float | None,
    seed: int,
) -> np.ndarray | None:
    if data_frac is None:
        return None

    data_frac = float(data_frac)
    assert 0.0 < data_frac <= 1.0, "data_frac must be in the interval (0, 1]"

    subset_size = int(np.ceil(num_records * data_frac))
    subset_size = max(1, subset_size)
    if subset_size >= num_records:
        return None

    rng = np.random.default_rng(seed)
    indices = rng.choice(num_records, size=subset_size, replace=False)
    return np.sort(indices).astype(np.int64)


def _ensure_grain_worker_flags_accessible() -> None:
    """Work around Grain worker profiling flag access before absl parsing."""
    from absl import flags

    flag_name = "grain_enable_multiprocess_worker_profiling"
    try:
        flags.FLAGS[flag_name].present = True
    except KeyError:
        pass


def _run_is_complete(run_dir: Path) -> bool:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return True

    text = meta_path.read_text()
    try:
        import yaml

        metadata = yaml.safe_load(text) or {}
    except Exception:
        metadata = _parse_simple_yaml(text)

    if not isinstance(metadata, dict) or "complete" not in metadata:
        return True
    return _coerce_bool(metadata["complete"])


def _parse_simple_yaml(text: str) -> dict[str, str]:
    metadata = {}
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            metadata[key.strip()] = value.strip()
    return metadata


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "yes", "1"}
    return bool(value)
