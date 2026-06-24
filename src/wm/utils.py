import json
from pathlib import Path

import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image

from wm.vae import VAE


def resize_img(image, shape: tuple[int, int] = (64, 64)):
    return np.asarray(Image.fromarray(image).resize(shape, Image.Resampling.BILINEAR))


def save_model(model: nnx.Module, model_name: str) -> None:
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir = Path("../checkpoints").resolve()
    fp = ckpt_dir / model_name
    checkpointer.save(fp, state)
    print(f"Saved model to {fp}")


def load_vae_checkpoint(run_dir, step=None):
    run_dir = Path(run_dir).resolve()

    with (run_dir / "config.json").open() as f:
        cfg = json.load(f)

    latent_dim = cfg["latent_dim"]
    seed = cfg["seed"]

    model = VAE(latent_dim=latent_dim, rngs=nnx.Rngs(seed))
    _, model_state = nnx.split(model)

    with ocp.CheckpointManager(run_dir) as manager:
        step = manager.latest_step() if step is None else step

        try:
            # Current train_vae.py checkpoints: {"model": ..., "optim": ...}
            tx = optax.adam(cfg.get("learning_rate", 1e-3))
            optim = nnx.Optimizer(model, tx, wrt=nnx.Param)
            _, optim_state = nnx.split(optim)

            restored = manager.restore(
                step,
                args=ocp.args.StandardRestore(
                    {"model": model_state, "optim": optim_state}
                ),
            )
            model_state = restored["model"]
        except Exception:
            # Older model-only checkpoints.
            model_state = manager.restore(
                step,
                args=ocp.args.StandardRestore(model_state),
            )

    nnx.update(model, model_state)
    return model, step
