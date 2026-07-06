import functools

import jax


# custom initialiser for more exploration under random policies
def cauchy_initializer(stddev: float):
    """Uses a heavy-tailed Cauchy distribution for weight initialization, with the
    goal of training more exploratory policies from random initialisations. As
    David Ha put it in his codebase - 'make it spicy'
    """

    def cauchy_init(
        key: jax.Array,
        shape: tuple[int, ...],
        dtype: jax.typing.DTypeLike,
        stddev: float,
    ) -> jax.Array:
        return jax.random.cauchy(key, shape, dtype) * stddev

    return functools.partial(cauchy_init, stddev=stddev)
