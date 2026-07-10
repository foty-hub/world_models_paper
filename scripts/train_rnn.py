from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx
from jaxtyping import Array, Shaped
from tqdm import tqdm

from wm import RNN
from wm.data import RNNSource, get_rnn_dataloader


def plot_loss(losses: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Negative log-likelihood (nats)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sample_latent(
    rngs: nnx.Rngs,
    mus: Shaped[Array, "B T Z"],
    logvars: Shaped[Array, "B T Z"],
) -> Shaped[Array, "B T Z"]:
    sigma = jnp.exp(0.5 * logvars)
    eps = rngs.normal(mus.shape, dtype=mus.dtype)
    return mus + sigma * eps


@nnx.value_and_grad
def loss_fn(
    model: RNN,
    z: Shaped[Array, "B T+1 Z"],
    acts: Shaped[Array, "B T A"],
) -> Shaped[Array, ""]:
    return -model.logprobs(z, acts).mean()


@nnx.jit
def train_step(
    model: RNN,
    optim: nnx.Optimizer,
    batch: dict[str, Array],
    latent_rngs: nnx.Rngs,
) -> Shaped[Array, ""]:
    z = sample_latent(latent_rngs, batch["mus"], batch["logvars"])
    loss, grads = loss_fn(model, z, batch["acts"])
    optim.update(model, grads)
    return loss


def main(
    data_dir: str = "data/random_data",
    vae_name: str = "vae",
    output_path: str = "experiments/rnn",
    batch_size: int = 32,
    num_epochs: int = 1,
    seed: int = 0,
    worker_count: int = 0,
    learning_rate: float = 2e-3,
    initializer_stddev: float = 1e-4,
) -> None:
    source = RNNSource(data_dir, vae_name)
    loader = get_rnn_dataloader(
        data_dir,
        vae_name=vae_name,
        batch_size=batch_size,
        num_workers=worker_count,
        num_epochs=num_epochs,
        seed=seed,
    )

    model = RNN(
        latent_dim=source.latent_dim,
        action_dim=source.action_dim,
        initializer_stddev=initializer_stddev,
        rngs=nnx.Rngs(seed),
    )
    tx = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate),
    )
    optim = nnx.Optimizer(model, tx, wrt=nnx.Param)
    latent_rngs = nnx.Rngs(seed)

    num_steps = len(source) // batch_size * num_epochs
    if num_steps == 0:
        raise ValueError("batch_size is larger than the number of episodes")

    losses: list[float] = []
    progress = tqdm(loader, total=num_steps, desc="Training RNN")
    for batch in progress:
        loss = train_step(model, optim, batch, latent_rngs)
        loss_value = float(np.asarray(loss))
        losses.append(loss_value)
        progress.set_postfix(loss=f"{loss_value:.3f}")

    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _, state = nnx.split(model)
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(output, state, force=True)
    print(f"Saved model to {output}")

    loss_path = output / "loss.png"
    plot_loss(losses, loss_path)
    print(f"Saved loss curve to {loss_path}")


if __name__ == "__main__":
    import tyro

    tyro.cli(main)
