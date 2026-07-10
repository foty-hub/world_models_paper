import chex
import distrax
import jax
import jax.numpy as jnp
from einops import rearrange, repeat
from flax import nnx
from flax.typing import Initializer
from jaxtyping import Array, Shaped

type Carry = tuple[
    Shaped[Array, "Batch RNNHiddenDim"],
    Shaped[Array, "Batch RNNHiddenDim"],
]


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
        kernel_init: Initializer | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.latent_dim = nnx.static(latent_dim)
        self.action_dim = nnx.static(action_dim)
        self.n_mixtures = nnx.static(n_mixtures)

        if kernel_init is None:
            cell = nnx.OptimizedLSTMCell(
                latent_dim + action_dim,
                hidden_units,
                rngs=rngs,
            )
            output_kernel_init = nnx.initializers.lecun_normal()
        else:
            cell = nnx.OptimizedLSTMCell(
                latent_dim + action_dim,
                hidden_units,
                rngs=rngs,
                kernel_init=kernel_init,
                recurrent_kernel_init=kernel_init,
            )
            output_kernel_init = kernel_init
        self.rnn = nnx.RNN(cell)
        # linear layer with outputs for GMM weights, means and logvars
        self.linear = nnx.Linear(
            hidden_units,
            n_mixtures * 2 * latent_dim + n_mixtures,
            rngs=rngs,
            kernel_init=output_kernel_init,
        )

    def __call__(
        self, x: Shaped[Array, "Batch Time Latent+ActionDim"], temperature: float = 1.0
    ) -> MDNRNNOut:
        chex.assert_rank(x, 3)
        x = self.rnn(x)  # type: ignore
        return self._stats_from_hidden(x, temperature)

    def _stats_from_hidden(
        self,
        hidden: Shaped[Array, "Batch Time RNNHiddenDim"],
        temperature: float,
    ) -> MDNRNNOut:
        """Project recurrent outputs into mixture-density parameters."""
        x = self.linear(hidden)

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
        z: Shaped[Array, "Batch Time+1 LatentDim"],
        actions: Shaped[Array, "Batch Time ActionDim"],
    ) -> Shaped[Array, "Batch Time"]:
        """Compute log p(z[t + 1] | z[t], action[t]) for each transition."""
        chex.assert_rank([z, actions], 3)
        chex.assert_axis_dimension(z, 2, self.latent_dim)
        chex.assert_axis_dimension(actions, 2, self.action_dim)
        if z.shape[0] != actions.shape[0] or z.shape[1] != actions.shape[1] + 1:
            raise ValueError("z must have shape [B, T + 1, Z] for actions [B, T, A]")

        x_t = jnp.concatenate([z[:, :-1], actions], axis=-1)
        z_t1 = z[:, 1:]
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

    def sample_step(
        self,
        z: Shaped[Array, "Batch LatentDim"],
        action: Shaped[Array, "Batch ActionDim"],
        carry: Carry,
        temperature: float = 1.0,
        key: Array | None = None,
    ) -> tuple[Shaped[Array, "Batch LatentDim"], Carry]:
        """Sample ``z[t + 1]`` and advance the recurrent state by one step.

        This is the autoregressive inference counterpart to :meth:`logprobs`:
        the current latent and action are consumed exactly once, and the returned
        carry must be supplied to the next call.
        """
        chex.assert_rank([z, action], 2)
        chex.assert_axis_dimension(z, 1, self.latent_dim)
        chex.assert_axis_dimension(action, 1, self.action_dim)
        if z.shape[0] != action.shape[0]:
            raise ValueError("z and action must have the same batch size")
        # Keep the eager API friendly while allowing ``temperature`` to become a
        # tracer when callers wrap this method in ``nnx.jit``.
        if isinstance(temperature, float) and temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        rnn_input = jnp.concatenate([z, action], axis=-1)
        carry, hidden = self.rnn.cell(carry, rnn_input)  # type: ignore
        hidden = rearrange(hidden, "B H -> B 1 H")
        stats = self._stats_from_hidden(hidden, temperature)

        # Reuse sequence sampling for the singleton time dimension. Sampling from
        # the already-computed stats avoids advancing the LSTM a second time.
        mu = stats.mu
        sigma = stats.sigma
        log_weights = jnp.log(stats.mixture_weights)
        batch_size, _, _, latent_dim = mu.shape

        if key is not None:
            noise_key, mixture_key = jax.random.split(key)
            indices = jax.random.categorical(mixture_key, log_weights, axis=-1)
            eps = jax.random.normal(
                noise_key, shape=(batch_size, 1, latent_dim), dtype=mu.dtype
            )
        else:
            indices = self.rngs.categorical(log_weights, axis=-1)
            eps = self.rngs.normal(shape=(batch_size, 1, latent_dim), dtype=mu.dtype)

        indices = rearrange(indices, "B T -> B T 1 1")
        selected_mu = jnp.take_along_axis(mu, indices, axis=-2)
        selected_sigma = jnp.take_along_axis(sigma, indices, axis=-2)
        selected_mu = rearrange(selected_mu, "B 1 1 Z -> B Z")
        selected_sigma = rearrange(selected_sigma, "B 1 1 Z -> B Z")
        eps = rearrange(eps, "B 1 Z -> B Z")

        z_next = selected_mu + jnp.sqrt(temperature) * selected_sigma * eps
        return z_next, carry

    def step(
        self,
        x: Shaped[Array, "Batch LatentDim+ActionDim"],
        carry: Carry,
    ) -> Carry:
        "Unroll one step of the hidden state without predicting a latent."
        x = rearrange(x, "B L -> B 1 L")
        # 2nd return item is the hidden states for each step - but we're only
        # considering 1 step so it's the same as the hidden state inside the carry
        carry, _ = self.rnn(x, return_carry=True, initial_carry=carry)
        return carry  # contains (y, h)

    def initialize_carry(self, num_envs):
        shape = (num_envs, self.latent_dim + self.action_dim)
        return self.rnn.cell.initialize_carry(shape, self.rngs)  # type: ignore
