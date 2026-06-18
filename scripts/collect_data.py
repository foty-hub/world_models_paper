import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import numpy as np
import yaml
from tqdm import tqdm

from wm.utils import resize_img

ROOT = (Path(__file__).resolve().parents[1] / "data").resolve()
ENV_ID = "CarRacing-v3"

N = 8  # parallel envs
T = 100_000  # total timesteps; make divisible by N


def write_meta(run_dir: Path, d):
    with open(run_dir / "meta.yaml", "w") as f:
        yaml.safe_dump(d, f, sort_keys=False)


def prep_obs(state):
    return np.asarray(resize_img(state[:84]), dtype=np.uint8)  # crop status bar


def collect_data(root: Path = ROOT, num_envs: int = N, total_timesteps: int = T):
    if total_timesteps % num_envs != 0:
        raise ValueError("total_timesteps must be divisible by num_envs")

    root.mkdir(parents=True, exist_ok=True)
    run_idx = len(list(root.glob("run_*")))
    run_dir = (root / f"run_{run_idx:04d}").resolve()
    run_dir.mkdir()
    print(f"Writing to {run_dir}")

    seg = total_timesteps // num_envs
    envs = gym.make_vec(ENV_ID, num_envs=num_envs, vectorization_mode="async")

    try:  # in a try-finally to close the envs.
        # Vec env distributes seeds like n, n+1, n+2, so advance by the
        # previous run's env count to avoid overlap.
        if run_idx == 0:
            seed = 0
        else:
            with open(root / f"run_{run_idx - 1:04d}" / "meta.yaml") as f:
                prev_meta = yaml.safe_load(f)
            seed = prev_meta["seed"] + prev_meta["num_envs"]

        o_batch, _ = envs.reset(seed=seed)
        proc_shape = prep_obs(o_batch[0]).shape
        act_shape = envs.single_action_space.shape

        meta = {
            "complete": False,
            "env_id": ENV_ID,
            "num_envs": num_envs,
            "total_timesteps": total_timesteps,
            "policy": "random",  # update when you swap in a real policy
            "seed": seed,
            "obs": {
                "shape": list(proc_shape),
                "dtype": "uint8",
                "preprocessing": "crop rows [0:84] (status bar), resize",
            },
            "act": {"shape": list(act_shape), "dtype": "float32"},
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        write_meta(run_dir, meta)  # exists from the start, marked incomplete
        t0 = time.time()

        # fmt: off
        obs   = np.lib.format.open_memmap(run_dir /   "obs.npy", mode="w+", dtype=np.uint8,   shape=(total_timesteps, *proc_shape))
        acts  = np.lib.format.open_memmap(run_dir /  "acts.npy", mode="w+", dtype=np.float32, shape=(total_timesteps, *act_shape))
        rews  = np.lib.format.open_memmap(run_dir /  "rews.npy", mode="w+", dtype=np.float32, shape=(total_timesteps,))
        dones = np.lib.format.open_memmap(run_dir / "dones.npy", mode="w+", dtype=bool,       shape=(total_timesteps,))
        # fmt: on

        cursor = np.arange(num_envs) * seg  # per-env write position
        seg_end = cursor + seg
        ep_starts = [[c] for c in cursor]  # first episode of each segment
        prev_done = np.zeros(num_envs, dtype=bool)

        with tqdm(total=total_timesteps) as pbar:
            while (cursor < seg_end).any():
                a = envs.action_space.sample()  # sample random action
                o_next, r, term, trunc, _ = envs.step(a)
                done = term | trunc

                for j in range(num_envs):
                    if cursor[j] >= seg_end[j]:
                        continue  # this env's segment is full; keep stepping, discard
                    if prev_done[j]:
                        # autoreset step: o_batch[j] was the final obs (already recorded),
                        # this step's action was ignored; record nothing, mark new episode.
                        ep_starts[j].append(cursor[j])
                    else:
                        c = cursor[j]
                        obs[c] = prep_obs(o_batch[j])
                        acts[c] = a[j]
                        rews[c] = r[j]
                        dones[c] = done[j]
                        cursor[j] += 1
                        pbar.update(1)

                o_batch = o_next
                prev_done = done

        for arr in (obs, acts, rews, dones):
            arr.flush()

        # merge: per-segment starts are already in global coordinates; sentinel = T
        bounds = np.array(
            sorted(s for starts in ep_starts for s in starts) + [total_timesteps]
        )
        np.save(run_dir / "ep_starts.npy", bounds)

        ep_lengths = np.diff(bounds)
        # per-episode returns from the flat reward array
        ep_returns = np.array(
            [rews[s:e].sum() for s, e in zip(bounds[:-1], bounds[1:])]
        )

        meta.update(
            {
                "complete": True,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "wall_time_sec": round(time.time() - t0, 1),
                "stats": {
                    "num_episodes": int(len(ep_lengths)),
                    "ep_length": {
                        "mean": float(ep_lengths.mean()),
                        "min": int(ep_lengths.min()),
                        "max": int(ep_lengths.max()),
                    },
                    "ep_return": {
                        "mean": float(ep_returns.mean()),
                        "std": float(ep_returns.std()),
                    },
                    "disk_bytes": int(
                        sum(
                            (run_dir / f"{n}.npy").stat().st_size
                            for n in ("obs", "acts", "rews", "dones")
                        )
                    ),
                },
            }
        )
        write_meta(run_dir, meta)

        print(f"Saved {int(len(ep_lengths))} rollouts to {run_dir}")
    finally:
        envs.close()


def main():
    parser = argparse.ArgumentParser(
        description=f"Collect random-policy rollouts from {ENV_ID}."
    )
    parser.add_argument(
        "--total-timesteps",
        "-t",
        type=int,
        default=T,
        help=f"Total number of timesteps to collect. Must be divisible by {N}.",
    )
    args = parser.parse_args()

    collect_data(total_timesteps=args.total_timesteps)


if __name__ == "__main__":
    main()
