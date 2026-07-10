import numpy as np
import pytest
import zarr

from wm.data import RNNSource, get_rnn_dataloader


def make_rnn_store(path, *, n_episodes=4, n_actions=3, latent_dim=2):
    z = zarr.create_group(path, overwrite=True)
    acts = np.arange(n_episodes * n_actions, dtype=np.float32).reshape(
        n_episodes, n_actions, 1
    )
    mus = np.zeros((n_episodes, n_actions + 1, latent_dim), dtype=np.float32)
    logvars = np.ones_like(mus)
    z.create_array("act", data=acts)
    z.create_array("latents/vae/mu", data=mus)
    z.create_array("latents/vae/logvar", data=logvars)
    return acts, mus, logvars


def test_rnn_source_returns_complete_episodes(tmp_path):
    acts, mus, logvars = make_rnn_store(tmp_path / "data")

    source = RNNSource(tmp_path / "data")
    episode = source[2]

    assert len(source) == 4
    assert source.action_dim == 1
    assert source.latent_dim == 2
    np.testing.assert_array_equal(episode["acts"], acts[2])
    np.testing.assert_array_equal(episode["mus"], mus[2])
    np.testing.assert_array_equal(episode["logvars"], logvars[2])


def test_rnn_dataloader_batches_episodes(tmp_path):
    make_rnn_store(tmp_path / "data")
    loader = get_rnn_dataloader(tmp_path / "data", batch_size=2)

    batch = next(iter(loader))

    assert batch["acts"].shape == (2, 3, 1)
    assert batch["mus"].shape == (2, 4, 2)
    assert batch["logvars"].shape == (2, 4, 2)


def test_rnn_source_rejects_wrong_transition_lengths(tmp_path):
    z = zarr.create_group(tmp_path / "data", overwrite=True)
    z.create_array("act", data=np.zeros((2, 3, 1), dtype=np.float32))
    z.create_array("latents/vae/mu", data=np.zeros((2, 3, 2), dtype=np.float32))
    z.create_array("latents/vae/logvar", data=np.zeros((2, 3, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="one more latent"):
        RNNSource(tmp_path / "data")
