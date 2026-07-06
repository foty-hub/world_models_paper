import cv2
import numpy as np
from jaxtyping import Float32, UInt8


# observation handling
def prep_obs(state: UInt8[np.ndarray, "... 96 96 3"]):
    "Crops out the status bar and resizes 96x96 car racing observations to 64x64"
    cropped = state[..., :84, :, :]
    return np.array([cv2.resize(i, (64, 64)) for i in cropped])


def normalise_obs(
    obs: UInt8[np.ndarray, "... 64 64 3"],
) -> Float32[np.ndarray, "... 64 64 3"]:
    "Takes uint8 observations, returns a floating point array bounded between [-1, +1]"
    return obs.astype(np.float32) / 127.5 - 1.0
