import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# Parameters
# ======================
m = 1.0           # Mass (kg)
L = 6.0           # Length (m) - Increased length
I = (1/12)*m*L**2 # Moment of inertia
g = 9.81          # Gravity (m/s²)
F_thrust = 70.0   # Thrust force (N)
dt = 0.001        # Smaller time step for quicker controller reaction
total_time = 10   # Simulation duration

# PID controller parameters
Kp = 2
Ki = 0.001
Kd = 0.3
initial_setpoint = np.deg2rad(69)  # Initial desired angle in radians
final_setpoint = np.deg2rad(105)   # Final desired angle in radians

# Initial state (pointing upward: theta0 = 90°)
state = np.array([0.0, 0.0, 0.0, 0.0, (np.pi/2), 0.0])  # [x, y, vx, vy, theta, omega]

# ======================
# Manual Euler Integration
# ======================
# Storage for results
times = np.arange(0, total_time, dt)
states = np.zeros((len(times), 6))
thrust_history = np.zeros(len(times))
error_history = np.zeros(len(times))
integral_error = 0.0
previous_error = 0.0

for i, t in enumerate(times):
    # Store current state
    states[i] = state
    x, y, vx, vy, theta, omega = state
    
    # Change setpoint halfway through flight
    if t < total_time / 2:
        desired_angle = initial_setpoint
    else:
        desired_angle = final_setpoint
    
    # PID controller to adjust thrust angle
    error = desired_angle - theta
    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    phi = - (Kp * error + Ki * integral_error + Kd * derivative_error)
    thrust_history[i] = phi
    error_history[i] = error
    previous_error = error
    
    # Calculate forces and torque
    Fx = F_thrust * np.cos(theta + phi)
    Fy = F_thrust * np.sin(theta + phi)
    torque = -(L/2) * F_thrust * np.sin(phi)
    
    # Update derivatives (Euler integration)
    new_state = [
        x + vx * dt,                     # x position
        y + vy * dt,                     # y position
        vx + (Fx/m) * dt,                # x velocity
        vy + (Fy/m - g) * dt,            # y velocity
        theta + omega * dt,              # angle
        omega + (torque/I) * dt          # angular velocity
    ]
    
    state = np.array(new_state)

# Extract results
x = states[:, 0]
y = states[:, 1]
theta = states[:, 4]

# ======================
# Animation
# ======================
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-1500, 1500)  # Adjusted to see the whole flight
ax.set_ylim(-5, 2000)   # Adjusted to see the whole flight
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Manual Rocket Simulation with PID Controller')
ax.grid(True)
ax.set_aspect('equal')  # Ensure equal scaling

rocket_line, = ax.plot([], [], 'r-', lw=3)  # Increased line width
thrust_vector, = ax.plot([], [], 'y-', lw=2)  # Increased line width
trail, = ax.plot([], [], 'b:', alpha=0.5)

def update(frame):
    # Current state
    x_curr = x[frame]
    y_curr = y[frame]
    theta_curr = theta[frame]
    phi_curr = thrust_history[frame]
    
    # Rocket body
    tip = [x_curr + (L/2)*np.cos(theta_curr), 
           y_curr + (L/2)*np.sin(theta_curr)]
    tail = [x_curr - (L/2)*np.cos(theta_curr), 
            y_curr - (L/2)*np.sin(theta_curr)]
    rocket_line.set_data([tail[0], tip[0]], [tail[1], tip[1]])
    
    # Thrust vector
    thrust_length = 150  # Increased thrust vector length
    thrust_vector.set_data(
        [x_curr, x_curr + thrust_length*np.cos(theta_curr + phi_curr)],
        [y_curr, y_curr + thrust_length*np.sin(theta_curr + phi_curr)]
    )
    
    # Trail
    trail.set_data(x[:frame], y[:frame])
    
    return rocket_line, thrust_vector, trail

ani = FuncAnimation(fig, update, frames=len(times), interval=dt*1000, blit=True)
plt.show()

# ======================
# Plot setpoint and error
# ======================
plt.figure(figsize=(10, 4))
plt.plot(times, np.rad2deg(error_history), label='Error (degrees)')
plt.axhline(y=0, color='r', linestyle='--', label='Setpoint (45 degrees)')
plt.axhline(y=np.rad2deg(final_setpoint - initial_setpoint), color='b', linestyle='--', label='Setpoint (-45 degrees)')
plt.xlabel('Time (s)')
plt.ylabel('Error (degrees)')
plt.title('Error vs Time')
plt.legend()
plt.grid(True)
plt.show()