import einops
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Shaped


# TODO: x inputs are uint8 - does that cause any issues?
class Encoder(nnx.Module):
    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.latent_dim = nnx.static(latent_dim)
        # fmt: off
        self.conv1 = nnx.Conv(  3,  32, (4, 4), strides=2, padding="VALID", rngs=self.rngs)
        self.conv2 = nnx.Conv( 32,  64, (4, 4), strides=2, padding="VALID", rngs=self.rngs)
        self.conv3 = nnx.Conv( 64, 128, (4, 4), strides=2, padding="VALID", rngs=self.rngs)
        self.conv4 = nnx.Conv(128, 256, (4, 4), strides=2, padding="VALID", rngs=self.rngs)
        # fmt: on
        self.dense = nnx.Linear(1024, self.latent_dim * 2, rngs=self.rngs)

    def __call__(self, x: Shaped[Array, "... H W C"]) -> Shaped[Array, "... LatentDim"]:
        "Batch-friendly encoder method."
        x = nnx.relu(self.conv1(x))
        x = nnx.relu(self.conv2(x))
        x = nnx.relu(self.conv3(x))
        x = nnx.relu(self.conv4(x))
        x = einops.rearrange(x, "... h w c -> ... (h w c)")  # flatten
        x = self.dense(x)

        # now we have a [32mu, 32sigma] vector -> convert to a latent
        mu, log_var = jnp.split(x, 2, axis=-1)
        sigma = jnp.exp(0.5 * log_var)  # ensure the standard deviation is positive

        # construct the latent vector as z = mu + sigma * N(0, 1)
        eps = self.rngs.normal(mu.shape, dtype=mu.dtype)

        # return everything so we can compute the KL divergence
        return {"mu": mu, "sigma": sigma, "z": mu + sigma * eps}


class Decoder(nnx.Module):
    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.latent_dim = nnx.static(latent_dim)
        self.dense = nnx.Linear(self.latent_dim, 1024, rngs=self.rngs)
        # fmt: off
        self.deconv1 = nnx.ConvTranspose(1024, 128, (5, 5), strides=2, padding="VALID", rngs=rngs)
        self.deconv2 = nnx.ConvTranspose( 128,  64, (5, 5), strides=2, padding="VALID", rngs=rngs)
        self.deconv3 = nnx.ConvTranspose(  64,  32, (6, 6), strides=2, padding="VALID", rngs=rngs)
        self.deconv4 = nnx.ConvTranspose(  32,   3, (6, 6), strides=2, padding="VALID", rngs=rngs)
        # fmt: on

    def __call__(self, z: Shaped[Array, "... LatentDim"]) -> Shaped[Array, "... H W C"]:
        z = nnx.relu(self.dense(z))
        z = einops.rearrange(z, "... L -> ... 1 1 L")  # unflatten for deconv layers
        z = nnx.relu(self.deconv1(z))
        z = nnx.relu(self.deconv2(z))
        z = nnx.relu(self.deconv3(z))
        z = nnx.sigmoid(self.deconv4(z))
        return z


class VAE(nnx.Module):
    def __init__(self, latent_dim: int, rngs: nnx.Rngs):
        self.rngs = rngs
        self.encoder = Encoder(latent_dim, self.rngs)
        self.decoder = Decoder(latent_dim, self.rngs)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
