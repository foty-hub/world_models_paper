import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from wm.vae import VAE


@nnx.value_and_grad
def overfit_loss(model: VAE, x: jnp.ndarray) -> jnp.ndarray:
    latent = model.encode(x)
    reconstruction = model.decode(latent.z)
    reconstruction_loss = 0.5 * jnp.mean(
        jnp.sum((x - reconstruction) ** 2, axis=(1, 2, 3))
    )
    kl_loss = -0.5 * jnp.mean(
        jnp.sum(
            1 + latent.logvar - latent.mu**2 - jnp.exp(latent.logvar), axis=-1
        )
    )
    return reconstruction_loss + model.beta * kl_loss


@nnx.jit
def overfit_step(model: VAE, optimizer: nnx.Optimizer, x: jnp.ndarray) -> None:
    loss, grads = overfit_loss(model, x)
    optimizer.update(model, grads)


def synthetic_images() -> jnp.ndarray:
    images = np.zeros((4, 64, 64, 3), dtype=np.float32)
    images[0, :32, :, 0] = 1.0
    images[1, 32:, :, 1] = 1.0
    images[2, :, :32, 2] = 1.0
    images[3, :, 32:, :] = 1.0
    return jnp.asarray(images)


def mean_reconstructions(model: VAE, x: jnp.ndarray) -> np.ndarray:
    return np.asarray(model.decode(model.encode(x).mu))


def test_vae_overfits_a_small_distinct_batch() -> None:
    images = synthetic_images()
    model = VAE(latent_dim=8, beta=0.1, rngs=nnx.Rngs(0))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

    initial_reconstructions = mean_reconstructions(model, images)
    initial_mse = np.mean((np.asarray(images) - initial_reconstructions) ** 2)

    for _ in range(50):
        overfit_step(model, optimizer, images)

    final_reconstructions = mean_reconstructions(model, images)
    final_mse = np.mean((np.asarray(images) - final_reconstructions) ** 2)
    reconstruction_variation = np.mean(np.var(final_reconstructions, axis=0))

    assert np.isfinite(final_reconstructions).all()
    assert final_mse < initial_mse * 0.25
    assert reconstruction_variation > 0.02
