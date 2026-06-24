import chex
import distrax
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import nnx
from jaxtyping import Array, Shaped


@chex.dataclass
class MDNRNNOut:
    mixture_weights: Shaped[Array, "B T N"]
    mu: Shaped[Array, "B T N Z"]
    sigma: Shaped[Array, "B T N Z"]


class MDNRNN(nnx.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        n_mixtures: int = 5,
        hidden_units: int = 256,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.latent_dim = nnx.static(latent_dim)
        self.action_dim = nnx.static(action_dim)
        self.n_mixtures = nnx.static(n_mixtures)

        self.rnn = nnx.RNN(
            nnx.OptimizedLSTMCell(latent_dim + action_dim, hidden_units, rngs=rngs)
        )
        # linear layer with outputs for GMM weights, means and logvars
        self.linear = nnx.Linear(
            hidden_units, n_mixtures * 2 * latent_dim + n_mixtures, rngs=rngs
        )

    def __call__(
        self, x: Shaped[Array, "Batch Time Latent+ActionDim"], temperature: float = 1.0
    ) -> MDNRNNOut:
        chex.assert_rank(x, 3)
        x = self.rnn(x)  # type: ignore
        x = self.linear(x)

        # Extract the GMM mixture weights
        weights: Shaped[Array, "B T N"] = x[..., : self.n_mixtures]
        weights = nnx.softmax(weights / temperature, axis=-1)  # weights should sum to 1

        # Extract the GMM means and log variances
        mu: Shaped[Array, "B T N*Z"]
        logvar: Shaped[Array, "B T N*Z"]
        mu, logvar = jnp.split(x[..., self.n_mixtures :], 2, axis=-1)
        sigma = jnp.exp(0.5 * logvar)  # Convert to positive std

        mu = rearrange(mu, "B T (N Z) -> B T N Z", Z=self.latent_dim)
        sigma = rearrange(sigma, "B T (N Z) -> B T N Z", Z=self.latent_dim)

        return MDNRNNOut(mixture_weights=weights, mu=mu, sigma=sigma)

    def logprobs(
        self,
        x: Shaped[Array, "Batch Time+1 Latent+ActionDim"],
    ) -> Shaped[Array, "Batch Time"]:
        "Given an input array x, call the MDN-RNN and compute the log probabilities of the data."
        chex.assert_axis_dimension(x, 2, self.latent_dim + self.action_dim)
        x_t = x[:, :-1, :]
        z_t1 = x[:, 1:, : self.latent_dim]
        z_t1 = repeat(z_t1, "B T Z -> B T N Z", N=self.n_mixtures)

        stats = self(x_t, 1.0)
        dist = distrax.MultivariateNormalDiag(stats.mu, stats.sigma)
        gmm_logprobs: Shaped[Array, "B T N"] = dist.log_prob(z_t1)  # type: ignore
        # computes log(sum_k w p_k(x))
        return nnx.logsumexp(a=gmm_logprobs, axis=-1, b=stats.mixture_weights)

    def sample(
        self,
        x: Shaped[Array, "Batch Time Latent+ActionDim"],
        temperature: float = 1.0,
        key: Array | None = None,
    ) -> Shaped[Array, "Batch Time LatentDim"]:
        stats = self(x, temperature)
        # fmt: off
        mu:      Shaped[Array, "B T N Z"] = stats.mu
        sigma:   Shaped[Array, "B T N Z"] = stats.sigma
        # need logits for weighted samples in random.categorical
        weights: Shaped[Array, "B T N"]   = jnp.log(stats.mixture_weights)
        # fmt: on

        B, T, N, Z = mu.shape
        noise_shape = (B, T, Z)

        # Generate latent samples
        # optionally take a key for lax.scan compatibility
        if key is not None:
            noise_key, gmm_key = jax.random.split(key)
            indices = jax.random.categorical(gmm_key, weights, axis=-1)
            eps = jax.random.normal(noise_key, shape=noise_shape)
        else:
            indices = self.rngs.categorical(weights, axis=-1)
            eps = self.rngs.normal(shape=noise_shape)

        indices = rearrange(indices, "B T -> B T 1 1")  # need same ndim for take

        # fmt: off
        # select GMM components
        mu:    Shaped[Array, "B T 1 Z"] = jnp.take_along_axis(mu   , indices, axis=-2)
        sigma: Shaped[Array, "B T 1 Z"] = jnp.take_along_axis(sigma, indices, axis=-2)
        # flatten
        mu    = rearrange(mu   , "B T 1 Z -> B T Z")
        sigma = rearrange(sigma, "B T 1 Z -> B T Z")

        # fmt: on

        # In Sketch-RNN, they define var -> temp * var. We're modifying std, so sqrt
        samples = mu + jnp.sqrt(temperature) * sigma * eps
        return samples

    def unroll(self, x_init: Shaped):
        "Autoregressively unroll over a given number of timesteps"
        ...
