"""Log likelihood lnp(V|I)
Includes code for both model and the likelihood function
"""

import jax.numpy as jnp
import jax
import jax.scipy as jsp


def model(uv: jnp.array, lm: jnp.array, image: jnp.array) -> jnp.complex64:
    """
    Model that returns
    :param uv: uv coordinates of the baseline, shape (2,) (Fourier frequencies of the image.)
    :param lm: lm coordinates of the image, can get from lm_arr = jnp.indices(im.shape). shape (2, npix, npix)
    :param image: the image. shape (npix, npix)
    :return: Visibility at the given uv coordinates.
    """
    l, m = lm
    u, v = uv
    return jnp.sum(image * jnp.exp(-2 * jnp.pi * 1j * (u * l + v * m)))


def lnprob(vis_obs: jnp.array, model_im: jnp.array, lm_arr: jnp.array, uv_arr: jnp.array, sigma: float) -> float:
    """
    Return the log likelihood.

    :param vis_obs: Observed visibility. Shape (n_baseline,
    :param model_im:
    :param lm_arr:
    :param uv_arr:
    :param sigma:
    :return:
    """
    vmap_model = jax.vmap(model, in_axes=(0, None, None))
    vis_model = vmap_model(uv_arr, model_im)
    return jnp.sum(jsp.stats.norm.logpdf(vis_obs, loc=vis_model, scale=sigma))
