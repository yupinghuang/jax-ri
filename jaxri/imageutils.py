"""Utility functions for dealing with images
"""

import numpy as np
import scipy
import jax.numpy as jnp
import jax.scipy as jsp

def cat_prior_image(image: jnp.array, sigma_noise: float = 0.1,
                    sigma_filter:float = 3, shift = (5, 10)) -> jnp.array:
  """
  Function to simulate sky observation prior using test image
  Args: image: jnp.array of test image
        sigma_noise: sigma for gaussian noise to be added
        sigma_filter: sigma to be used with gaussian smoothing
        shift: desired shift of the image
  Returns: jnp.array of sky observation prior
  """
  im = np.asarray(image) #convert image to numpy array
  im = scipy.ndimage.gaussian_filter(im, sigma=sigma_filter) #gaussian smoothing/blur
  noise = np.random.normal(scale = sigma_noise, size = image.shape) #gaussian noise
  im = im + noise
  im = scipy.ndimage.shift(im, shift = shift) #shift image
  
  return jnp.asarray(im)


def sky_observation_prior(image: jnp.array, sigma: float = 5,
                          shift = (5, 10)) -> jnp.array:
  """
  Function to simulate sky observation prior using test image
  Args: image: jnp.array of test image
        sigma: sigma for gaussian noise to be added
        shift: desired shift of the image
  Returns: jnp.array of sky observation prior
  """
  im = np.asarray(image) #convert jnp.array to numpy array
  im = scipy.ndimage.gaussian_filter(im, sigma=sigma) #convolution
  noise = np.random.normal(size = image.shape) #generate noise
  im = im + noise
  im = scipy.ndimage.shift(im, shift) #shift image
  idx_transient = np.random.randint(im.shape) 
  im[idx_transient[0], idx_transient[1]] = 20 #add a new source

  return jnp.asarray(im)


def mean_std_map(image: jnp.array, box_size : int = 15) -> jnp.array:
  """
  Function to compute rolling mean and standard deviation at each pixel over small windows 
  Args: image: the image to evaluate rolling mean and std over
        box_size: integer specifying the box size of the window
  Returns: mean map, std map
  """
  mean = scipy.ndimage.uniform_filter(image, (box_size, box_size))
  mean_sqr = scipy.ndimage.uniform_filter(image**2, (box_size, box_size))
  std = np.sqrt(mean_sqr - mean**2)

  return jnp.asarray(mean), jnp.asarray(std)
