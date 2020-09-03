"""Compute log priors
"""
import jax.numpy as jnp
import jax.scipy as jsp


def ln_ill_defined_prior(image: jnp.array) -> jnp.array:
    """
    This is equivalent to optimizing the likelihood function only. i.e. a "uniform over all real number" kind of
    prior, which mathematically does not exist.

    :param image: the image to evaluate the prior on.
    :return:
    """
    return jnp.ones(shape=image.shape)


def ln_some_other_prior(image: jnp.array) -> jnp.array:
    # jsp.stats.norm.logpdf for a Gaussian log PDF?
    # jsp.stats.laplace.logpdf for a Gaussian log PDF?
    # A prior that enforces positive sky value?
    return


"""
Note: if you're doing Gaussian prior, your parameters will be different for different test cases. JAX does not like
branching (if clauses) inside functions, so our best bet is to have one ln_norm_prior for each test case (say
log_norm_cat_prior, ln_norm_sky_prior.

You can generate the Gaussian prior that encodes previous observations by taking a rolling mean and std across a
modified version of the ground truth image that represents a previous observation.
You can generate such an image by blurring the groud-truth image with a convolution (simulating a lower-res obsrvation),
or by adding significant noises (less sensitivity), or by not including one of the sources (the sky being variable
itself or missing spatial scales.)
"""