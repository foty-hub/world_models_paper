# Now to train the controller - in the original paper it's a single-layer model that takes h_t, a_t, z_t
import chex
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Shaped

from .controller import Controller
from .rnn import MDNRNN, Carry
from .vae import VAE


# TODO: will there be weirdness about randomness from different trace levels? TBD
class Agent(nnx.Module):
    def __init__(
        self,
        vae: VAE | None = None,
        rnn: MDNRNN | None = None,
        controller: Controller | None = None,
        latent_dim: int = 32,
        action_dim: int = 3,
        *,
        rngs: nnx.Rngs,
    ):
        self.vae = vae if vae else VAE(latent_dim, rngs=rngs)
        self.rnn = rnn if rnn else MDNRNN(latent_dim, action_dim, rngs=rngs)
        self.controller = controller if controller else Controller(rngs=rngs)

    def initialize_carry(self, num_envs: int) -> Carry:
        return self.rnn.initialize_carry(num_envs)

    @nnx.jit
    def __call__(
        self,
        obs: Shaped[Array, "Batch H W C"],
        carry: Carry,
    ) -> tuple[Shaped[Array, "Batch ActionDim"], Carry]:
        # Extract the hidden state from the carry. NNX's RNN returns
        # a carry like rnn(x) -> (y, h)
        chex.assert_rank(obs, 4)
        chex.assert_rank(carry[0], 2)
        _, h = carry
        latent = self.vae.encode(obs).z

        action = self.controller(jnp.concatenate([latent, h], axis=-1))

        rnn_in = jnp.concatenate([latent, action], axis=-1)
        carry = self.rnn.step(rnn_in, carry)
        return action, carry
