import json
from pathlib import Path

import optax
import orbax.checkpoint as ocp
from flax import nnx

from wm.rnn import MDNRNN
from wm.vae import VAE


# model checkpointing utils
def save_model(model: nnx.Module, model_name: str) -> None:
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir = Path("../checkpoints").resolve()
    fp = ckpt_dir / model_name
    checkpointer.save(fp, state)
    print(f"Saved model to {fp}")


def load_rnn(
    model_name: str,
    latent_dim: int = 32,
    action_dim: int = 3,
    n_mixtures: int = 5,
    hidden_units: int = 256,
    rngs: nnx.Rngs | None = None,
) -> MDNRNN:
    rngs = rngs if rngs else nnx.Rngs(0)
    model = MDNRNN(
        latent_dim,
        action_dim,
        n_mixtures=n_mixtures,
        hidden_units=hidden_units,
        rngs=rngs,
    )
    _, state = nnx.split(model)

    checkpointer = ocp.StandardCheckpointer()
    state = checkpointer.restore(Path(model_name).resolve(), target=state)
    nnx.update(model, state)
    return model


def load_vae(run_dir, step=None, rngs: nnx.Rngs | None = None):
    run_dir = Path(run_dir).resolve()

    with (run_dir / "config.json").open() as f:
        cfg = json.load(f)

    latent_dim = cfg["latent_dim"]
    seed = cfg["seed"]

    rngs = rngs if rngs else nnx.Rngs(seed)
    model = VAE(latent_dim=latent_dim, rngs=rngs)
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
