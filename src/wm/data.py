from os import PathLike

import grain.python as grain
import numpy as np
import zarr
from jaxtyping import Shaped


class VAESource(grain.RandomAccessDataSource):
    def __init__(self, path: str | PathLike):
        z = zarr.open_group(path, mode="r")
        self.obs: zarr.Array = z["obs"]  # type: ignore

        shape = self.obs.shape
        self.n_episodes = shape[0]
        self.n_timesteps = shape[1]
        self.n_samples = self.n_episodes * self.n_timesteps

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Shaped[np.ndarray, "H W C"]:
        # We use divmod here as the inverse equivalent of...
        #   ix = ep_ix * num_timesteps + time_ix
        ep_idx, time_idx = divmod(idx, self.n_timesteps)
        return np.asarray(self.obs[ep_idx, time_idx])


def get_vae_dataloader(
    path: str | PathLike,
    frac: float = 1.0,
    batch_size: int = 64,
    num_workers: int = 0,
    num_epochs: int = 1,
) -> grain.DataLoader:
    """Returns a grain dataloader for"""
    source = VAESource(path)

    n_records = int(len(source) * frac)  # not ideal, but good enough for this
    sampler = grain.IndexSampler(
        num_records=n_records, shuffle=True, num_epochs=num_epochs, seed=0
    )
    loader = grain.DataLoader(
        data_source=source,
        sampler=sampler,
        worker_count=num_workers,
        operations=[grain.BatchOperation(batch_size, drop_remainder=True)],
    )

    return loader
