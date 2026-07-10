from pathlib import Path

import jax
import numpy as np
import zarr
from einops import rearrange
from flax import nnx
from jaxtyping import Array, Shaped, UInt8
from tqdm import trange

from wm.utils import load_vae, normalise_obs

run_name = "vae"
data_dir = Path("data") / "random_data"


ckpt_dir = Path("experiments") / run_name
model, _ = load_vae(ckpt_dir)
model.eval()
graphdef, state = nnx.split(model)

z = zarr.open_group(data_dir)
obs_dset: zarr.Array = z["obs"]  # type: ignore (shape - [B T H W C])
num_records = obs_dset.shape[0]

# fmt: off
print("Creating arrays")
mus     = z.create_array(f"latents/{run_name}/mu",     shape=(0, 1001, 32), chunks=(1, 1001, 32), dtype=np.float32, overwrite=True)
logvars = z.create_array(f"latents/{run_name}/logvar", shape=(0, 1001, 32), chunks=(1, 1001, 32), dtype=np.float32, overwrite=True)
# fmt: on


@jax.jit
def get_latents(
    state: nnx.State,
    x: Shaped[Array, "B H W C"],
) -> tuple[Shaped[Array, "B Z"], Shaped[Array, "B Z"]]:
    model = nnx.merge(graphdef, state)
    latents = model.encode(x)
    return latents.mu, latents.logvar


# Construct the latent datasets we'll use to train the RNN -
#  1. Iterate over each episode in the dataset
#  2. Compute the VAE's latent parameters (mu, logvar)
#  3. Save them to the datasets. Note we don't record the specific sample z
#      because we'll resample it during RNN training
for i in trange(0, num_records):
    batch: UInt8[np.ndarray, "T H W C"] = obs_dset[i]  # type: ignore
    batch = normalise_obs(batch)

    mu, logvar = get_latents(state, batch)  # Shape [T, Z]

    # When saving, restore the batch dimension so we can access entire episodes for
    # RNN training
    mus.append(rearrange(mu, "T Z -> 1 T Z"))
    logvars.append(rearrange(logvar, "T Z -> 1 T Z"))
