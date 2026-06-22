import json
from datetime import datetime, timezone
from pathlib import Path
from pprint import pprint

import einops
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jaxtyping import Array, Shaped

from wm.data import make_observation_loader
from wm.vae import VAE


def plot_reconstruction(
    x: Shaped[Array, "B H W C"],
    model: VAE,
    fp: str | Path,
    *,
    n_plots: int = 5,
) -> None:
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    n_plots = min(n_plots, x.shape[0])
    x = x[:n_plots]
    x_pred = model(x)

    x = np.clip(np.asarray(x), 0.0, 1.0)
    x_pred = np.clip(np.asarray(x_pred), 0.0, 1.0)

    fig, axes = plt.subplots(n_plots, 2, figsize=(4, 2 * n_plots), squeeze=False)
    for i in range(n_plots):
        axes[i, 0].imshow(x[i])
        axes[i, 0].axis("off")
        axes[i, 1].imshow(x_pred[i])
        axes[i, 1].axis("off")

    axes[0, 0].set_title("Real")
    axes[0, 1].set_title("Reconstructed")
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


@nnx.value_and_grad(has_aux=True)
def loss_fn(model: VAE, x: Shaped[Array, "B H W C"]) -> Shaped[Array, ""]:
    latent_dict = model.encode(x)
    mu, logvar, z = latent_dict.to_tuple()

    x_pred = model.decode(z)

    decoder_sigma = model.sigma

    D = 64 * 64 * 3  # number of pixels/channels
    err = x - x_pred
    mse = einops.einsum(err, err, "B h w c, B h w c -> B")
    recon_loss = (
        mse / (2 * decoder_sigma**2)
        + D * jnp.log(decoder_sigma)
        + D / 2 * jnp.log(2 * jnp.pi)
    )
    recon_loss = jnp.mean(recon_loss)
    kl_loss = -0.5 * jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=-1)
    kl_loss = jnp.mean(kl_loss)

    loss = recon_loss + kl_loss
    return loss, {"loss": loss, "kl": kl_loss, "recon": recon_loss}


@nnx.jit
def train_step(model, optim, batch):
    (_, loss), grads = loss_fn(model, batch)
    optim.update(model, grads)
    return loss


def plot_loss(losses: list[dict[str, float]], fp: str | Path) -> None:
    fp = Path(fp)
    fp.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    steps = [d["step"] for d in losses]
    ax.plot(steps, [d["loss"] for d in losses], label="Loss")
    ax.plot(steps, [d["kl"] for d in losses], label="KL")
    ax.plot(steps, [d["recon"] for d in losses], label="Reconstruction")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fp)
    plt.close(fig)


def to_float_metrics(metrics: dict[str, Array], step: int) -> dict[str, float]:
    return {"step": step, **{k: float(np.asarray(v)) for k, v in metrics.items()}}


def save_config(run_dir: Path, config: dict, resume_step: int | None) -> None:
    config_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "resume_step": resume_step,
        "config": config,
    }

    config_fp = run_dir / "config.json"
    with config_fp.open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")

    history_fp = run_dir / "config_history.jsonl"
    with history_fp.open("a") as f:
        json.dump(config_record, f)
        f.write("\n")


def load_losses(run_dir: Path) -> list[dict[str, float]]:
    losses_fp = run_dir / "losses.jsonl"
    if not losses_fp.exists():
        return []

    losses = []
    with losses_fp.open() as f:
        for line in f:
            line = line.strip()
            if line:
                losses.append(json.loads(line))
    return losses


def append_loss(run_dir: Path, loss: dict[str, float]) -> None:
    losses_fp = run_dir / "losses.jsonl"
    with losses_fp.open("a") as f:
        json.dump(loss, f)
        f.write("\n")


def log_training(
    model: VAE,
    losses: list[dict[str, float]],
    reconstruction_batch: Shaped[Array, "B H W C"],
    run_dir: Path,
    step: int,
) -> None:
    plot_loss(losses, run_dir / "loss.png")
    plot_reconstruction(
        reconstruction_batch,
        model,
        run_dir / "reconstructions" / f"step_{step:06d}.png",
    )


def checkpoint_model(
    model: nnx.Module,
    optim: nnx.Optimizer,
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
) -> None:
    _, model_state = nnx.split(model)
    _, optim_state = nnx.split(optim)
    checkpoint_manager.save(
        step,
        args=ocp.args.StandardSave({"model": model_state, "optim": optim_state}),
    )


def restore_checkpoint(
    model: nnx.Module,
    optim: nnx.Optimizer,
    checkpoint_manager: ocp.CheckpointManager,
    step: int,
) -> bool:
    _, model_state = nnx.split(model)
    _, optim_state = nnx.split(optim)

    try:
        restored = checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore({"model": model_state, "optim": optim_state}),
        )
    except Exception:
        restored_model = checkpoint_manager.restore(
            step,
            args=ocp.args.StandardRestore(model_state),
        )
        nnx.update(model, restored_model)
        return False

    nnx.update(model, restored["model"])
    nnx.update(optim, restored["optim"])
    return True


# CLI args
# --batch-size
# --seed
# --num-epochs
# --worker-count
# --data-frac
# --latent-dim
# --learning-rate
# --log-every
# --checkpoint-every
# --run-name
# --data-dir
# --experiments-dir
# --max-checkpoints-to-keep
def main(
    run_name: str = "vae_train",
    batch_size: int = 128,
    seed: int = 0,
    num_epochs: int = 1,
    worker_count: int = 2,
    data_frac: float = 0.2,
    latent_dim: int = 32,
    learning_rate: float = 8e-4,
    log_every: int = 500,
    ckpt_every: int = 1_000,
    data_dir: str = "data",
    experiments_dir: str = "experiments",
    max_checkpoints_to_keep: int | None = 3,
):
    config = locals()
    print("Config:")
    pprint(config, sort_dicts=False)

    run_dir = Path(experiments_dir) / run_name
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_options = ocp.CheckpointManagerOptions(
        max_to_keep=max_checkpoints_to_keep,
        create=True,
    )
    with ocp.CheckpointManager(
        run_dir,
        options=checkpoint_options,
    ) as checkpoint_manager:
        latest_step = checkpoint_manager.latest_step()
        save_config(run_dir, config, latest_step)

        loader = make_observation_loader(
            data_dir,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            num_epochs=num_epochs,
            worker_count=worker_count,
            data_frac=data_frac,
        )

        n_examples = len(loader._data_source)
        print(f"Loaded {n_examples:,} examples in loader.")

        model = VAE(latent_dim=latent_dim, rngs=nnx.Rngs(seed))

        # TODO: learning rate schedule goes here
        tx = optax.adam(learning_rate=learning_rate)
        optim = nnx.Optimizer(model, tx, wrt=nnx.Param)

        if latest_step is not None:
            restored_optim = restore_checkpoint(
                model,
                optim,
                checkpoint_manager,
                latest_step,
            )
            msg = f"Restored checkpoint at step {latest_step}"
            if restored_optim:
                msg += " with optimizer state"
            else:
                msg += " without optimizer state"
            print(f"{msg}.")

        # Extract one batch that we'll reuse for every time we log.
        reconstruction_batch = next(iter(loader))

        losses = load_losses(run_dir)
        last_checkpoint_step = latest_step
        start_step = latest_step + 1 if latest_step is not None else 0

        print(f"Beginning training over {n_examples // batch_size:,} training steps")
        for local_step, x in enumerate(loader):
            global_step = start_step + local_step
            loss = train_step(model, optim, x)
            loss_record = to_float_metrics(loss, global_step)
            losses.append(loss_record)
            append_loss(run_dir, loss_record)

            # Book-keeping
            if global_step % log_every == 0:
                log_training(model, losses, reconstruction_batch, run_dir, global_step)
                print(f"Wrote logs to {run_dir}")
            if global_step % ckpt_every == 0:
                checkpoint_model(model, optim, checkpoint_manager, global_step)
                last_checkpoint_step = global_step
                print(f"Queued checkpoint for step {global_step} in {run_dir}")

        if losses:
            final_step = int(losses[-1]["step"])
            if final_step != last_checkpoint_step:
                checkpoint_model(model, optim, checkpoint_manager, final_step)
                print(f"Queued final checkpoint for step {final_step} in {run_dir}")
            log_training(model, losses, reconstruction_batch, run_dir, final_step)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
