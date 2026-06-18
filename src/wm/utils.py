from pathlib import Path

import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from PIL import Image


def resize_img(image, shape: tuple[int, int] = (64, 64)):
    return np.asarray(Image.fromarray(image).resize(shape, Image.Resampling.BILINEAR))


def save_model(model: nnx.Module, model_name: str) -> None:
    _, state = nnx.split(model)
    checkpointer = ocp.StandardCheckpointer()
    ckpt_dir = Path("../checkpoints").resolve()
    fp = ckpt_dir / model_name
    checkpointer.save(fp, state)
    print(f"Saved model to {fp}")


def load_model(model_class: nnx.Module): ...
