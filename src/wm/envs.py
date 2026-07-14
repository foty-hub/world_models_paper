"""Project-specific Gymnasium environments."""

import pygame
from gymnasium.envs.box2d.car_racing import (
    GRASS_DIM,
    PLAYFIELD,
    SCALE,
    STATE_H,
    STATE_W,
    WINDOW_H,
    WINDOW_W,
    ZOOM,
    CarRacing,
)
from gymnasium.envs.registration import EnvSpec


_GRASS_POLYGONS = tuple(
    (
        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y),
        (GRASS_DIM * x, GRASS_DIM * y),
        (GRASS_DIM * x, GRASS_DIM * y + GRASS_DIM),
        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
    )
    for x in range(-20, 20, 2)
    for y in range(-20, 20, 2)
)


class ControllerTrainingCarRacing(CarRacing):
    """CarRacing's state-pixel renderer without the discarded dashboard.

    Controller training crops the 96x96 observation to its first 84 rows before
    resizing it to 64x64. Gymnasium draws the dashboard and reward text entirely
    within the discarded rows, so omitting them preserves the model input exactly.

    Track and grass coordinates are also cached, while camera transforms, clipping,
    antialiasing, and mutable road-tile colours are recomputed on every frame.
    Physics, rewards, termination, and the returned 96x96 observation shape remain
    unchanged.
    """

    def _render(self, mode: str):
        # Keep the upstream renderer for interactive use. Training calls
        # ``_render("state_pixels")`` from CarRacing.step.
        if mode != "state_pixels":
            return super()._render(mode)

        if "t" not in self.__dict__:
            return None

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        angle = -self.car.hull.angle
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -self.car.hull.position[0] * zoom
        scroll_y = -self.car.hull.position[1] * zoom
        translation = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        translation = (
            WINDOW_W / 2 + translation[0],
            WINDOW_H / 4 + translation[1],
        )

        self._render_road(zoom, translation, angle)
        self.car.draw(self.surf, zoom, translation, angle, False)
        self.surf = pygame.transform.flip(self.surf, False, True)

        # Deliberately retain Gymnasium's 96x96 output. prep_obs still performs
        # the 96x84 crop followed by the slightly squashed 64x64 resize.
        return self._create_image_array(self.surf, (STATE_W, STATE_H))

    def _render_road(self, zoom, translation, angle) -> None:
        # CarRacing replaces road_poly whenever it generates a new track. Cache
        # only its immutable coordinates and rebuild when that object changes.
        if getattr(self, "_cached_road_source", None) is not self.road_poly:
            self._cached_road_source = self.road_poly
            self._cached_road_polygons = tuple(
                tuple((point[0], point[1]) for point in polygon)
                for polygon, _ in self.road_poly
            )

        bounds = PLAYFIELD
        field = (
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        )
        self._draw_colored_polygon(
            self.surf,
            field,
            self.bg_color,
            zoom,
            translation,
            angle,
            clip=False,
        )

        for polygon in _GRASS_POLYGONS:
            self._draw_colored_polygon(
                self.surf,
                polygon,
                self.grass_color,
                zoom,
                translation,
                angle,
            )

        # Tile colours change when the car visits them, so they must remain live.
        for polygon, (_, color) in zip(
            self._cached_road_polygons, self.road_poly, strict=True
        ):
            self._draw_colored_polygon(
                self.surf,
                polygon,
                [int(channel) for channel in color],
                zoom,
                translation,
                angle,
            )


CONTROLLER_TRAINING_ENV_SPEC = EnvSpec(
    id="ControllerTrainingCarRacing-v0",
    entry_point="wm.envs:ControllerTrainingCarRacing",
    reward_threshold=900,
    max_episode_steps=1000,
)
