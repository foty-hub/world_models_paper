import jax.numpy as jnp
from flax import nnx
from flax.typing import Initializer
from jaxtyping import Array, Shaped


class Controller(nnx.Module):
    def __init__(
        self,
        rnn_hidden_dim: int = 256,
        latent_dim: int = 32,
        action_dim: int = 3,
        kernel_init: Initializer | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        in_dim = rnn_hidden_dim + latent_dim
        kernel_init = kernel_init or nnx.initializers.lecun_normal()
        self.linear = nnx.Linear(
            in_features=in_dim,
            out_features=action_dim,
            rngs=rngs,
            kernel_init=kernel_init,
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
