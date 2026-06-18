from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from wm.data import ObservationDataSource, make_observation_loader


class ObservationDataSourceTest(unittest.TestCase):
    def test_indexes_observations_across_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = np.arange(12, dtype=np.uint8).reshape(3, 2, 2, 1)
            second = (100 + np.arange(8, dtype=np.uint8)).reshape(2, 2, 2, 1)
            _write_run(root, 0, first)
            _write_run(root, 1, second)

            source = ObservationDataSource(root, normalize=False)

            self.assertEqual(len(source), 5)
            np.testing.assert_array_equal(source[0], first[0])
            np.testing.assert_array_equal(source[2], first[2])
            np.testing.assert_array_equal(source[3], second[0])
            np.testing.assert_array_equal(source[4], second[1])

    def test_normalizes_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obs = np.array([[[[0], [255]]]], dtype=np.uint8)
            _write_run(root, 0, obs)

            sample = ObservationDataSource(root)[0]

            self.assertEqual(sample.dtype, np.float32)
            np.testing.assert_allclose(sample, obs[0].astype(np.float32) / 255.0)

    def test_skips_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            incomplete = np.full((2, 2, 2, 1), 7, dtype=np.uint8)
            complete = np.full((1, 2, 2, 1), 9, dtype=np.uint8)
            _write_run(root, 0, incomplete, complete=False)
            _write_run(root, 1, complete)

            source = ObservationDataSource(root, normalize=False)

            self.assertEqual(len(source), 1)
            np.testing.assert_array_equal(source[0], complete[0])

    def test_loader_batches_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obs = np.arange(16, dtype=np.uint8).reshape(4, 2, 2, 1)
            _write_run(root, 0, obs)

            loader = make_observation_loader(
                root,
                batch_size=3,
                shuffle=False,
                num_epochs=1,
                drop_remainder=False,
            )
            batches = list(loader)

            self.assertEqual(len(batches), 2)
            self.assertEqual(batches[0].shape, (3, 2, 2, 1))
            self.assertEqual(batches[0].dtype, np.float32)
            np.testing.assert_allclose(
                batches[0], obs[:3].astype(np.float32) / 255.0
            )
            self.assertEqual(batches[1].shape, (1, 2, 2, 1))


def _write_run(
    root: Path,
    index: int,
    obs: np.ndarray,
    *,
    complete: bool = True,
):
    run_dir = root / f"run_{index:04d}"
    run_dir.mkdir()
    np.save(run_dir / "obs.npy", obs)
    run_dir.joinpath("meta.yaml").write_text(
        f"complete: {str(complete).lower()}\n",
    )


if __name__ == "__main__":
    unittest.main()

