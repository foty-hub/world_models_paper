import jax.numpy as jnp
import numpy as np
from flax import nnx

from wm import Agent
from wm.controller import Controller
from wm.rnn import MDNRNN
from wm.vae import VAE


INITIALIZER_VALUE = 0.125


def constant_initializer(key, shape, dtype=jnp.float32, out_sharding=None):
    del key, out_sharding
    return jnp.full(shape, INITIALIZER_VALUE, dtype)


def assert_custom_kernel_and_default_bias(layer) -> None:
    np.testing.assert_array_equal(
        np.asarray(layer.kernel),
        np.full(layer.kernel.shape, INITIALIZER_VALUE, dtype=layer.kernel.dtype),
    )
    if layer.bias is not None:
        np.testing.assert_array_equal(
            np.asarray(layer.bias), np.zeros(layer.bias.shape, dtype=layer.bias.dtype)
        )


def test_models_use_flax_initializers_by_default() -> None:
    vae = VAE(latent_dim=4, rngs=nnx.Rngs(0))
    rnn = MDNRNN(
        latent_dim=4,
        action_dim=2,
        n_mixtures=2,
        hidden_units=8,
        rngs=nnx.Rngs(1),
    )
    controller = Controller(
        latent_dim=4,
        rnn_hidden_dim=8,
        action_dim=2,
        rngs=nnx.Rngs(2),
    )
    assert isinstance(rnn.rnn.cell, nnx.OptimizedLSTMCell)
    rnn_cell = rnn.rnn.cell

    for layer in (
        vae.encoder.conv1,
        vae.encoder.dense,
        vae.decoder.dense,
        vae.decoder.deconv4,
        rnn_cell.dense_i,
        rnn_cell.dense_h,
        rnn.linear,
        controller.linear,
    ):
        assert np.isfinite(np.asarray(layer.kernel)).all()
        assert not np.all(np.asarray(layer.kernel) == INITIALIZER_VALUE)
        if layer.bias is not None:
            np.testing.assert_array_equal(
                np.asarray(layer.bias),
                np.zeros(layer.bias.shape, dtype=layer.bias.dtype),
            )


def test_agent_propagates_custom_initializer_to_every_kernel() -> None:
    agent = Agent(
        latent_dim=4,
        action_dim=2,
        kernel_init=constant_initializer,
        rngs=nnx.Rngs(0),
    )
    assert isinstance(agent.rnn.rnn.cell, nnx.OptimizedLSTMCell)
    rnn_cell = agent.rnn.rnn.cell

    vae_layers = (
        agent.vae.encoder.conv1,
        agent.vae.encoder.conv2,
        agent.vae.encoder.conv3,
        agent.vae.encoder.conv4,
        agent.vae.encoder.dense,
        agent.vae.decoder.dense,
        agent.vae.decoder.deconv1,
        agent.vae.decoder.deconv2,
        agent.vae.decoder.deconv3,
        agent.vae.decoder.deconv4,
    )
    rnn_and_controller_layers = (
        rnn_cell.dense_i,
        rnn_cell.dense_h,
        agent.rnn.linear,
        agent.controller.linear,
    )

    for layer in (*vae_layers, *rnn_and_controller_layers):
        assert_custom_kernel_and_default_bias(layer)
