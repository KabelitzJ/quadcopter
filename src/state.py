import numpy as np

from parameters import parameters

def A_linearized():
  g = parameters["g"]

  # Taken from https://uu.diva-portal.org/smash/get/diva2:1870673/FULLTEXT01.pdf

  return np.array([
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, g, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -g, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ])

def B_linearized():
  m = parameters["m"]
  I_x = parameters["I_x"]
  I_y = parameters["I_y"]
  I_z = parameters["I_z"]

  # Taken from https://uu.diva-portal.org/smash/get/diva2:1870673/FULLTEXT01.pdf

  return np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [1/m, 0, 0, 0],
    [0, 1/I_x, 0, 0],
    [0, 0, 1/I_y, 0],
    [0, 0, 0, 1/I_z],
  ])

def C_linearized():
  return np.eye(6, 12)

def D_linearized():
  return np.zeros((6, 4))
