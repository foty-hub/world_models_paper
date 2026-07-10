import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from wm.rnn import MDNRNN


def make_model():
    return MDNRNN(
        latent_dim=4,
        action_dim=2,
        n_mixtures=2,
        hidden_units=8,
        rngs=nnx.Rngs(0),
    )


def test_logprobs_scores_every_transition():
    model = make_model()
    z = jnp.ones((2, 6, 4))  # [B T+1 Z]
    actions = jnp.ones((2, 5, 2))  # [B T A]

    logprobs = model.logprobs(z, actions)

    assert logprobs.shape == (2, 5)
    assert np.isfinite(np.asarray(logprobs)).all()


def test_logprobs_rejects_misaligned_sequences():
    model = make_model()

    with pytest.raises(ValueError, match=r"\[B, T \+ 1, Z\]"):
        model.logprobs(jnp.ones((2, 5, 4)), jnp.ones((2, 5, 2)))
