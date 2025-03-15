import numpy as np
import control as ctrl

class System:

  def __init__(self, A, B, C, D):
    self.A = A
    self.B = B
    self.C = C
    self.D = D

  def is_controllable(self):
    C0 = ctrl.ctrb(self.A, self.B)
    return np.linalg.matrix_rank(C0) == self.A.shape[0]
  
  def is_observable(self):
    B0 = ctrl.obsv(self.A, self.C)
    return np.linalg.matrix_rank(B0) == self.A.shape[0]

  def x_dot(self, x, u):
    return self.A @ x + self.B @ u
  
  def y(self, x, u):
    return self.C @ x + self.D @ u
  
  def lqr(self, Q, R):
    K, _, _ =  ctrl.lqr(self.A, self.B, Q, R)
    return K
