import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Shaped

from .initializer import cauchy_initializer


class Controller(nnx.Module):
    def __init__(
        self,
        rnn_hidden_dim: int = 256,
        latent_dim: int = 32,
        action_dim: int = 3,
        initializer_stddev: float = 0.01,
        *,
        rngs: nnx.Rngs,
    ):
        # NNX's default initializer is LeCunNormal, but Ha & Schmidhuber uses a Cauchy
        init = cauchy_initializer(initializer_stddev)
        in_dim = rnn_hidden_dim + latent_dim
        self.linear = nnx.Linear(
            in_features=in_dim,
            out_features=action_dim,
            rngs=rngs,
            kernel_init=init,
            bias_init=init,
        )

    def __call__(
        self, x: Shaped[Array, "... HiddenDim+LatentDim+ActionDim"]
    ) -> Shaped[Array, "... ActionDim"]:
        a = self.linear(x)
        a = nnx.tanh(a)  # clamp to [-1, +1]

        # fmt: off
                                                                 # steer: [-1, +1]
        a = a.at[..., 1].set((a[..., 1] / 2.0 + 0.5))            #   gas: [ 0,  1]
        a = a.at[..., 2].set(jnp.clip(a[..., 2], 0.0, 1.0))      # brake: [ 0,  1], zero at neutral
        # fmt: on
        return a
