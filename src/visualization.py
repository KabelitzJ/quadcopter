import numpy as np
from parameters import parameters
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Button

def rotation_matrix(roll, pitch, yaw):
  """ Returns a 3D rotation matrix from roll (φ), pitch (θ), and yaw (ψ). """
  R_x = np.array([[1, 0, 0],
                  [0, np.cos(roll), -np.sin(roll)],
                  [0, np.sin(roll), np.cos(roll)]])
  
  R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                  [0, 1, 0],
                  [-np.sin(pitch), 0, np.cos(pitch)]])
  
  R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]])
  
  return R_z @ R_y @ R_x  # Final rotation order: Yaw → Pitch → Roll

import matplotlib.pyplot as plt

def visualization2d(t, result, fn):
  fig, axs = plt.subplots(1, 2, figsize=(18, 10))

  axs[0].plot(t, result.y[0, :], label="x")
  axs[0].plot(t, result.y[1, :], label="y")
  axs[0].plot(t, result.y[2, :], label="z")
  axs[0].plot(t, fn(t), label="z_target", linestyle="--")
  axs[0].legend()
  axs[0].set_xlabel("Time [s]")  # Correct method
  axs[0].set_ylabel("Position [m]")  # Correct method
  axs[0].set_title("Position over time")  # Correct method
  axs[0].grid()

  axs[1].plot(t, result.y[3, :], label="φ")
  axs[1].plot(t, result.y[4, :], label="θ")
  axs[1].plot(t, result.y[5, :], label="ψ")
  axs[1].legend()
  axs[1].set_xlabel("Time [s]")  # Correct method
  axs[1].set_ylabel("Orientation [rad]")  # Correct method
  axs[1].set_title("Orientation over time")  # Correct method
  axs[1].grid()

  plt.tight_layout()
  plt.show()


def visualization3d(t, result):
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')

  # Extract position and orientation
  x = result.y[0] # x
  y = result.y[1] # y
  z = result.y[2] # z
  roll = result.y[3] # φ
  pitch = result.y[4] # θ
  yaw = result.y[5] # ψ

  # Define quadcopter arm length
  arm_length = parameters["L"]

  # Initialize quadcopter body lines
  line1, = ax.plot([], [], [], 'b-', lw=2)  # One arm
  line2, = ax.plot([], [], [], 'r-', lw=2)  # Other arm
  trace, = ax.plot([], [], [], 'g--', lw=1)  # Flight path

  # Set plot limits and labels
  ax.set_xlim([-2, 2])
  ax.set_ylim([-2, 2])
  ax.set_zlim([0, 2])
  ax.set_xlabel('X [m]')
  ax.set_ylabel('Y [m]')
  ax.set_zlabel('Z [m]')
  ax.set_title("Quadcopter 3D Simulation")

  def init():
    line1.set_data([], [])
    line1.set_3d_properties([])

    line2.set_data([], [])
    line2.set_3d_properties([])

    trace.set_data([], [])
    trace.set_3d_properties([])

    return line1, line2, trace

  def update(frame):
    # Current position
    x_pos = x[frame]
    y_pos = y[frame]
    z_pos = z[frame]

    # Current orientation
    R = rotation_matrix(roll[frame], pitch[frame], yaw[frame])

    # Define quadcopter arms in local coordinates
    arm1_local = np.array([[-arm_length, 0, 0], [arm_length, 0, 0]]).T
    arm2_local = np.array([[0, -arm_length, 0], [0, arm_length, 0]]).T

    # Rotate arms
    arm1_rotated = R @ arm1_local
    arm2_rotated = R @ arm2_local

    # Translate to world position
    arm1_x = arm1_rotated[0] + x_pos
    arm1_y = arm1_rotated[1] + y_pos
    arm1_z = arm1_rotated[2] + z_pos

    arm2_x = arm2_rotated[0] + x_pos
    arm2_y = arm2_rotated[1] + y_pos
    arm2_z = arm2_rotated[2] + z_pos

    # Update plot
    line1.set_data(arm1_x, arm1_y)
    line1.set_3d_properties(arm1_z)
    
    line2.set_data(arm2_x, arm2_y)
    line2.set_3d_properties(arm2_z)

    trace.set_data(x[:frame], y[:frame])
    trace.set_3d_properties(z[:frame])

    return line1, line2, trace

  ani = animation.FuncAnimation(fig, update, frames=len(t), init_func=init, interval=20, blit=True)

  # Create buttons to control animation
  ax_reset = plt.axes([0.8, 0.02, 0.1, 0.05])  # Reset button position
  button_reset = Button(ax_reset, 'Reset')

  ax_start = plt.axes([0.8, 0.09, 0.1, 0.05])  # Start button position
  button_start = Button(ax_start, 'Start')

  ax_stop = plt.axes([0.8, 0.16, 0.1, 0.05])  # Stop button position
  button_stop = Button(ax_stop, 'Stop')

  def reset_animation(event):
      ani.event_source.stop()  # Stop current animation
      ani.frame_seq = ani.new_frame_seq()  # Reset animation frame sequence
      ani.event_source.start()  # Restart animation

  def start_animation(event):
      ani.event_source.start()  # Start animation

  def stop_animation(event):
      ani.event_source.stop()  # Stop animation

  button_reset.on_clicked(reset_animation)
  button_start.on_clicked(start_animation)
  button_stop.on_clicked(stop_animation)

  # Initially pause the animation
  ani.event_source.stop()

  plt.show()
