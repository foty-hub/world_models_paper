from flax import nnx
from jaxtyping import Array, Shaped


class Controller(nnx.Module):
    def __init__(
        self,
        rnn_hidden_dim: int = 256,
        latent_dim: int = 32,
        action_dim: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        in_dim = rnn_hidden_dim + latent_dim
        self.linear = nnx.Linear(in_features=in_dim, out_features=action_dim, rngs=rngs)

    def __call__(
        self, x: Shaped[Array, "... HiddenDim+LatentDim+ActionDim"]
    ) -> Shaped[Array, "... ActionDim"]:
        a = self.linear(x)
        a = nnx.tanh(a)  # clamp to [-1, +1]
        # squash action components into [-1:1, 0:1, 0:1]
        a = a.at[..., 1:].set(0.5 + a[..., 1:] / 2)
        return a
