"""Log likelihood lnp(V|I)
Includes code for both model and the likelihood function
"""

import jax.numpy as jnp
import jax.scipy as jsp

# noise per baseline
sigma = 1.


def model(uv: jnp.array, im: jnp.array) -> float:
    """
    Model that returns
    :param uv: uv coordinates of the baseline, shape (2,)
    :param im: the image
    :return: Visibility at the given uv coordinates.
    """
    return 0.


def lnprob(vis_obs: jnp.array, model_im: jnp.array, uv_arr: jnp.array) -> float:
    # vis_model = vmap(model)(uv_arr, im)
    # return jnp.sum(jsp.stats.norm.logpdf(vis_model - vis_obs), sigma))
    pass