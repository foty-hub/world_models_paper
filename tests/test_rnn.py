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


def test_sample_step_advances_carry_and_returns_next_latent():
    model = make_model()
    z = jnp.ones((2, 4))
    actions = jnp.zeros((2, 2))
    carry = model.initialize_carry(2)

    z_next, next_carry = model.sample_step(
        z, actions, carry, temperature=0.8, key=jnp.array([0, 1], dtype=jnp.uint32)
    )

    assert z_next.shape == (2, 4)
    assert next_carry[0].shape == (2, 8)
    assert next_carry[1].shape == (2, 8)
    assert np.isfinite(np.asarray(z_next)).all()
    assert not np.array_equal(np.asarray(next_carry[0]), np.asarray(carry[0]))


def test_sample_step_is_reproducible_with_explicit_key():
    model = make_model()
    z = jnp.ones((1, 4))
    actions = jnp.zeros((1, 2))
    carry = model.initialize_carry(1)
    key = jnp.array([7, 11], dtype=jnp.uint32)

    first, first_carry = model.sample_step(z, actions, carry, key=key)
    second, second_carry = model.sample_step(z, actions, carry, key=key)

    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_carry[0], second_carry[0])
    np.testing.assert_array_equal(first_carry[1], second_carry[1])


def test_sample_step_rejects_non_positive_temperature():
    model = make_model()
    carry = model.initialize_carry(1)

    with pytest.raises(ValueError, match="temperature"):
        model.sample_step(jnp.ones((1, 4)), jnp.zeros((1, 2)), carry, 0.0)
