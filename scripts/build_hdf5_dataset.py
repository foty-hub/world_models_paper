from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import tyro
from tqdm import tqdm


DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUTPUT = Path("data/world_models.h5")


def build_hdf5_dataset(
    data_dir: str | PathLike[str] = DEFAULT_DATA_DIR,
    output: str | PathLike[str] = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
    compression: str | None = "lzf",
    chunk_size: int = 1024,
) -> Path:
    data_dir = Path(data_dir).expanduser()
    output = Path(output).expanduser()
    compression = None if compression == "none" else compression

    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        output.unlink()

    runs = _load_runs(data_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_frames = runs[-1]["end"]
    print(f"Found {len(runs)} runs with {total_frames:,} frames.")
    print(f"Writing {output}...")

    with h5py.File(output, "w") as h5:
        datasets = _create_datasets(
            h5,
            runs,
            chunk_size=chunk_size,
            compression=compression,
        )
        _write_attrs(h5, data_dir, runs)
        _copy_runs(
            datasets,
            runs,
            chunk_size=chunk_size,
        )

    return output


def _load_runs(data_dir: Path) -> list[dict]:
    runs = []
    offset = 0

    for run_dir in _discover_run_dirs(data_dir):
        obs = np.load(run_dir / "obs.npy", mmap_mode="r")
        run = {
            "path": run_dir,
            "name": run_dir.name,
            "start": offset,
            "end": offset + obs.shape[0],
            "obs": obs,
            "acts": np.load(run_dir / "acts.npy", mmap_mode="r"),
            "rews": np.load(run_dir / "rews.npy", mmap_mode="r"),
            "dones": np.load(run_dir / "dones.npy", mmap_mode="r"),
            "ep_starts": np.load(run_dir / "ep_starts.npy"),
            "meta": (run_dir / "meta.yaml").read_text(),
        }
        runs.append(run)
        offset = run["end"]

    return runs


def _discover_run_dirs(data_dir: Path) -> list[Path]:
    return [
        run_dir
        for run_dir in sorted(data_dir.glob("run_*"))
        if run_dir.is_dir() and _run_is_complete(run_dir)
    ]


def _run_is_complete(run_dir: Path) -> bool:
    meta_path = run_dir / "meta.yaml"
    if not meta_path.exists():
        return True

    for line in meta_path.read_text().splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "complete":
            return value.strip().lower() in {"true", "yes", "1"}
    return True


def _create_datasets(
    h5: h5py.File,
    runs: list[dict],
    *,
    chunk_size: int,
    compression: str | None,
) -> dict[str, h5py.Dataset]:
    first = runs[0]
    total_frames = runs[-1]["end"]
    num_episodes = sum(run["ep_starts"].shape[0] - 1 for run in runs)
    frame_chunk = min(chunk_size, total_frames)

    obs = first["obs"]
    acts = first["acts"]
    rews = first["rews"]
    dones = first["dones"]

    datasets = {
        "observations": h5.create_dataset(
            "observations",
            shape=(total_frames, *obs.shape[1:]),
            dtype=obs.dtype,
            chunks=(frame_chunk, *obs.shape[1:]),
            compression=compression,
        ),
        "actions": h5.create_dataset(
            "actions",
            shape=(total_frames, *acts.shape[1:]),
            dtype=acts.dtype,
            chunks=(frame_chunk, *acts.shape[1:]),
        ),
        "rewards": h5.create_dataset(
            "rewards",
            shape=(total_frames,),
            dtype=rews.dtype,
            chunks=(frame_chunk,),
        ),
        "dones": h5.create_dataset(
            "dones",
            shape=(total_frames,),
            dtype=dones.dtype,
            chunks=(frame_chunk,),
        ),
    }

    datasets["observations"].attrs["source_dtype"] = str(obs.dtype)
    datasets["actions"].attrs["source_dtype"] = str(acts.dtype)

    episodes = h5.create_group("episodes")
    datasets["episodes/start"] = episodes.create_dataset(
        "start",
        shape=(num_episodes,),
        dtype=np.int64,
    )
    datasets["episodes/end"] = episodes.create_dataset(
        "end",
        shape=(num_episodes,),
        dtype=np.int64,
    )

    runs_group = h5.create_group("runs")
    string_dtype = h5py.string_dtype(encoding="utf-8")
    datasets["runs/name"] = runs_group.create_dataset(
        "name",
        shape=(len(runs),),
        dtype=string_dtype,
    )
    datasets["runs/start"] = runs_group.create_dataset(
        "start",
        shape=(len(runs),),
        dtype=np.int64,
    )
    datasets["runs/end"] = runs_group.create_dataset(
        "end",
        shape=(len(runs),),
        dtype=np.int64,
    )

    h5.create_group("run_metadata")
    return datasets


def _write_attrs(h5: h5py.File, data_dir: Path, runs: list[dict]) -> None:
    h5.attrs["format"] = "world_models_hdf5_v1"
    h5.attrs["created_at"] = datetime.now(timezone.utc).isoformat()
    h5.attrs["source_data_dir"] = str(data_dir)
    h5.attrs["total_frames"] = runs[-1]["end"]
    h5.attrs["num_runs"] = len(runs)
    h5.attrs["num_episodes"] = sum(run["ep_starts"].shape[0] - 1 for run in runs)


def _copy_runs(
    datasets: dict[str, h5py.Dataset],
    runs: list[dict],
    *,
    chunk_size: int,
) -> None:
    episode_cursor = 0
    run_metadata = datasets["observations"].file["run_metadata"]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    total_bytes = sum(
        run[name].nbytes
        for run in runs
        for name in ("obs", "acts", "rews", "dones")
    )

    progress = tqdm(
        total=total_bytes,
        desc="Copying arrays",
        unit="B",
        unit_scale=True,
    )
    with progress:
        for run_index, run in enumerate(runs):
            for source_name, destination_name in (
                ("obs", "observations"),
                ("acts", "actions"),
                ("rews", "rewards"),
                ("dones", "dones"),
            ):
                progress.set_postfix_str(f"{run['name']}:{destination_name}")
                _copy_array(
                    run[source_name],
                    datasets[destination_name],
                    run["start"],
                    chunk_size,
                    progress,
                )

            ep_starts = run["start"] + run["ep_starts"]
            num_episodes = ep_starts.shape[0] - 1
            episode_slice = slice(episode_cursor, episode_cursor + num_episodes)
            datasets["episodes/start"][episode_slice] = ep_starts[:-1]
            datasets["episodes/end"][episode_slice] = ep_starts[1:]
            episode_cursor += num_episodes

            datasets["runs/name"][run_index] = run["name"]
            datasets["runs/start"][run_index] = run["start"]
            datasets["runs/end"][run_index] = run["end"]
            run_metadata.create_dataset(
                run["name"],
                data=run["meta"],
                dtype=string_dtype,
            )


def _copy_array(
    source: np.ndarray,
    destination: h5py.Dataset,
    start: int,
    chunk_size: int,
    progress: tqdm,
) -> None:
    for local_start in range(0, source.shape[0], chunk_size):
        local_end = min(local_start + chunk_size, source.shape[0])
        chunk = source[local_start:local_end]
        destination[start + local_start : start + local_end] = chunk
        progress.update(chunk.nbytes)


def main(
    data_dir: Path = DEFAULT_DATA_DIR,
    output: Path = DEFAULT_OUTPUT,
    overwrite: bool = False,
    compression: str = "lzf",
    chunk_size: int = 1024,
) -> None:
    """Aggregate data/run_* NumPy rollouts into one HDF5 file."""
    output = build_hdf5_dataset(
        data_dir=data_dir,
        output=output,
        overwrite=overwrite,
        compression=compression,
        chunk_size=chunk_size,
    )
    print(f"Wrote HDF5 dataset to {output}")


if __name__ == "__main__":
    tyro.cli(main)
