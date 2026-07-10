"""Drive the learned CarRacing world model in a Pygame window."""

from pathlib import Path

import jax
import numpy as np
import pygame
import tyro
import zarr
from flax import nnx

from wm.imagination import (
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    TEMPERATURE_STEP,
    RolloutSelection,
    adjust_temperature,
    car_racing_action,
)
from wm.utils import load_rnn, load_vae, normalise_obs

IMAGE_SIZE = 64
STATUS_HEIGHT = 112
BACKGROUND = (18, 18, 22)
FOREGROUND = (235, 235, 240)
MUTED = (165, 165, 175)


@nnx.jit
def encode_mean(model, observation):
    return model.encode(observation).mu


@nnx.jit
def decode_latent(model, latent):
    return model.decode(latent)


@nnx.jit
def sample_step(model, latent, action, carry, temperature, key):
    return model.sample_step(latent, action, carry, temperature, key)


def validate_args(
    *, temperature: float, scale: int, fps: int, episode: int | None
) -> None:
    if not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE:
        raise ValueError(
            f"temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
        )
    if scale < 1:
        raise ValueError("scale must be at least 1")
    if fps < 1:
        raise ValueError("fps must be at least 1")
    if episode is not None and episode < 0:
        raise ValueError("episode must be non-negative")


def decoded_frame(vae, latent: jax.Array) -> np.ndarray:
    image = np.asarray(decode_latent(vae, latent)[0])
    if not np.isfinite(image).all():
        raise RuntimeError("VAE decoder produced a non-finite image")
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def draw_status(
    screen: pygame.Surface,
    font: pygame.font.Font,
    *,
    image_height: int,
    selection: RolloutSelection,
    step: int,
    action: np.ndarray,
    temperature: float,
    paused: bool,
) -> None:
    status = "PAUSED" if paused else "RUNNING"
    lines = [
        (
            f"{status}  episode {selection.episode}  step {step}  "
            f"temperature {temperature:.2f}",
            FOREGROUND,
        ),
        (
            f"action  steer {action[0]:+.1f}  gas {action[1]:.1f}  "
            f"brake {action[2]:.1f}",
            FOREGROUND,
        ),
        ("arrows drive | +/- temperature | space pause | . step", MUTED),
        ("R replay | N new seed frame | Esc quit", MUTED),
    ]
    for index, (line, colour) in enumerate(lines):
        text = font.render(line, True, colour)
        screen.blit(text, (10, image_height + 8 + index * 24))


def main(
    vae_path: str = "experiments/vae",
    rnn_path: str = "experiments/rnn",
    data_path: str = "data/random_data",
    episode: int | None = None,
    seed: int = 0,
    temperature: float = 1.0,
    scale: int = 8,
    fps: int = 50,
) -> None:
    """Interact with the trained MDN-RNN and VAE entirely in imagination."""
    validate_args(temperature=temperature, scale=scale, fps=fps, episode=episode)

    vae_dir = Path(vae_path).resolve()
    rnn_dir = Path(rnn_path).resolve()
    dataset_dir = Path(data_path).resolve()
    for label, path in (
        ("VAE checkpoint", vae_dir),
        ("RNN checkpoint", rnn_dir),
        ("dataset", dataset_dir),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    dataset = zarr.open_group(dataset_dir, mode="r")
    if "obs" not in dataset or "act" not in dataset:
        raise ValueError("dataset must contain 'obs' and 'act' arrays")
    observations = dataset["obs"]
    actions = dataset["act"]
    if observations.ndim != 5 or observations.shape[-3:] != (64, 64, 3):
        raise ValueError("dataset observations must have shape [E, T, 64, 64, 3]")
    if actions.ndim != 3 or actions.shape[-1] != 3:
        raise ValueError("dataset actions must have shape [E, T, 3]")
    if actions.shape[0] != observations.shape[0]:
        raise ValueError("dataset observations and actions must have equal episode counts")
    num_episodes = observations.shape[0]
    if episode is not None and episode >= num_episodes:
        raise ValueError(f"episode must be less than {num_episodes}")

    print(f"Loading VAE from {vae_dir} ...")
    vae, vae_step = load_vae(vae_dir)
    vae.eval()
    latent_dim = int(vae.encoder.latent_dim)
    print(f"Loading RNN from {rnn_dir} ...")
    rnn = load_rnn(rnn_dir, latent_dim=latent_dim, action_dim=3)
    rnn.eval()
    if int(rnn.latent_dim) != latent_dim or int(rnn.action_dim) != 3:
        raise ValueError("VAE, RNN, and CarRacing dimensions are incompatible")
    print(f"Loaded VAE step {vae_step}; starting imagination demo.")

    selection_rng = np.random.default_rng(seed)
    first_episode = (
        episode if episode is not None else int(selection_rng.integers(num_episodes))
    )
    selection = RolloutSelection(episode=first_episode, rollout_seed=seed)

    def reset(current: RolloutSelection):
        observation = np.asarray(observations[current.episode, 0])
        observation = normalise_obs(observation)[None]
        latent = encode_mean(vae, observation)
        carry = rnn.initialize_carry(1)
        key = jax.random.PRNGKey(current.rollout_seed)
        return latent, carry, key, decoded_frame(vae, latent)

    latent, carry, rollout_key, frame = reset(selection)
    step = 0
    paused = False
    single_step = False

    pygame.init()
    image_extent = IMAGE_SIZE * scale
    screen = pygame.display.set_mode((image_extent, image_extent + STATUS_HEIGHT))
    pygame.display.set_caption("World Models — CarRacing Imagination")
    font = pygame.font.Font(None, 22)
    clock = pygame.time.Clock()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_PERIOD and paused:
                        single_step = True
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        temperature = adjust_temperature(
                            temperature, TEMPERATURE_STEP
                        )
                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        temperature = adjust_temperature(
                            temperature, -TEMPERATURE_STEP
                        )
                    elif event.key == pygame.K_r:
                        latent, carry, rollout_key, frame = reset(selection)
                        step = 0
                    elif event.key == pygame.K_n:
                        selection = RolloutSelection(
                            episode=int(selection_rng.integers(num_episodes)),
                            rollout_seed=int(
                                selection_rng.integers(np.iinfo(np.uint32).max)
                            ),
                        )
                        latent, carry, rollout_key, frame = reset(selection)
                        step = 0

            keys = pygame.key.get_pressed()
            action = car_racing_action(
                left=keys[pygame.K_LEFT],
                right=keys[pygame.K_RIGHT],
                gas=keys[pygame.K_UP],
                brake=keys[pygame.K_DOWN],
            )

            if (not paused or single_step) and running:
                rollout_key, transition_key = jax.random.split(rollout_key)
                latent, carry = sample_step(
                    rnn,
                    latent,
                    action[None],
                    carry,
                    temperature,
                    transition_key,
                )
                if not np.isfinite(np.asarray(latent)).all():
                    raise RuntimeError("RNN produced a non-finite latent")
                frame = decoded_frame(vae, latent)
                step += 1
                single_step = False

            screen.fill(BACKGROUND)
            image_surface = pygame.image.frombuffer(
                frame.tobytes(), (IMAGE_SIZE, IMAGE_SIZE), "RGB"
            )
            image_surface = pygame.transform.scale(
                image_surface, (image_extent, image_extent)
            )
            screen.blit(image_surface, (0, 0))
            draw_status(
                screen,
                font,
                image_height=image_extent,
                selection=selection,
                step=step,
                action=action,
                temperature=temperature,
                paused=paused,
            )
            pygame.display.flip()
            clock.tick(fps)
    finally:
        pygame.quit()


if __name__ == "__main__":
    tyro.cli(main)
