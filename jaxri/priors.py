"""Compute log priors
"""
import jax.numpy as jnp
import jax.scipy as jsp

BIG_NUMBER = 10000

def ln_ill_defined_prior(image: jnp.array) -> jnp.array:
    """
    This is equivalent to optimizing the likelihood function only. 
    i.e. a "uniform over all real number" kind of
    prior, which mathematically does not exist.

    :param image: the image to evaluate the prior on.
    :return: 
    """
    return jnp.ones(shape=image.shape)

def ln_positive_number(image: jnp.array) -> jnp.array:
    return jsp.stats.uniform.logpdf(image, loc=0, scale=BIG_NUMBER)

def ln_positive_laplace(image: jnp.array, laplace_scale: float = 1) -> jnp.array:
  """
  Function to evaluate a laplacian prior with positive values
  Args: image: the image to evaluate the prior on
        laplace_scale: scale parameter to be passed on to jsp.stats.laplace.logpdf()
  Returns: log of laplacian distribution over positive numbers
  """
  lnpdf_laplace = jsp.stats.laplace.logpdf(image, loc = 0, scale = laplace_scale)
  lnpdf_uniform = jsp.stats.uniform.logpdf(image, loc=0, scale = 1000)
  ln_prior = lnpdf_laplace + lnpdf_uniform
  return ln_prior

def ln_gaussian_prior(image: jnp.array, mean: jnp.array, std: jnp.array) -> jnp.array:
  """
  Function to calculate gaussian prior over an image
  Args: image: the image to evaluate prior on
        mean: mean map obtained from mean_std_map()
        std: std map obtained from mean_std_map()
  Returns: log of gaussian prior evaluated over the image
  """
  return jsp.stats.norm.logpdf(image, loc=mean, scale = std)

