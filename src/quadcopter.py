import numpy as np
import control as ctrl
import matplotlib.pyplot as plt

from parameters import parameters
from visualization import visualization2d, visualization3d

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

def fn(x):
  return 1.7 * np.sin(1.7 * x)

def target(t, fn):
  target = np.zeros((12, len(t)))
  target[0, :] = 1.7 * np.cos(1.7 * t) + t * 0.25
  target[1, :] = 1.7 * np.sin(1.7 * t) + t * 0.25
  target[2, :] = t * 0.25
  return target

class Result:
  def __init__(self, t, x0):
    self.t = t
    self.x = x0

    self.y = np.zeros((12, len(self.t)))
    self.y[:, 0] = x0

    self.u = np.zeros((4, len(self.t)))
    self.u[:, 0] = np.zeros(4)

def main():
  # Define the system

  A = A_linearized()
  B = B_linearized()
  C = C_linearized()
  D = D_linearized()

  system = ctrl.ss(A, B, C, D)

  # Implement observer for the system

  # TODO

  # Implement LQR regulator for the system

  Q = np.diag([10, 10, 10, 1, 1, 1, 2, 2, 2, 0.5, 0.5, 0.5])
  R = np.diag([2, 2, 2, 2])

  K, _, _ = ctrl.lqr(system, Q, R)

  # Define the reference filter N (example filter)
  N = np.eye(12)

  # Create closed-loop system

  A_cl = A - B @ K

  system_cl = ctrl.ss(A_cl, B, C, D)

  # Simulate the system

  t = np.linspace(0, 60, 6000)
  
  x0 = np.zeros(12)
  # x0[0] = 0  # Initial x position
  # x0[1] = 4  # Initial y position
  # x0[2] = 1  # Initial z position
  x0[3] = 0.1  # Small roll angle
  u = np.zeros((4, t.size))
  r = target(t, fn)

  result = Result(t, x0)

  for i in range(1, len(t)):
    # Calculate the error between current state and target
    error = result.x - r[:, i-1]

    # Update control input using LQR
    u[:, i] = -K @ error  # Feedback control law

    # Update state (state-space dynamics)
    dx = A @ result.x + B @ u[:, i]  # System dynamics
    result.x = result.x + dx * (t[i] - t[i-1])  # Integrate state

    # Store the results
    # result.y[:, i] = C @ result.x + D @ u[:, i]
    result.y[:, i] = result.x
    result.u[:, i] = u[:, i]

  # Plot the results

  visualization2d(t, result, fn)
  visualization3d(t, result)

if __name__ == "__main__":
  main()
