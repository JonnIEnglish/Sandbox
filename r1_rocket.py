import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# Parameters
# ======================
m = 1.0          # Mass (kg)
L = 6.0           # Length (m) - Increased length
I = (1/12)*m*L**2 # Moment of inertia
g = 9.81          # Gravity (m/s²)
F_thrust = 60   # Thrust force (N)
dt = 0.001        # Smaller time step for quicker controller reaction
burn_time = 3    # Thrust duration
total_time = 30   # Maximum simulation duration

# PID controller parameters
Kp = 2
Ki = 0.001
Kd = 0.3
initial_setpoint = np.deg2rad(90)  # Initial desired angle in radians
final_setpoint = np.deg2rad(60)     # Final desired angle in radians

# Initial state (pointing upward: theta0 = 90°)
state = np.array([0.0, 0.0, 0.0, 0.0, (np.pi/2), 0.0])  # [x, y, vx, vy, theta, omega]

# ======================
# Manual Euler Integration
# ======================
# Storage for results
times = []
states = []
thrust_history = []
error_history = []
integral_error = 0.0
previous_error = 0.0
t = 0.0

while t < total_time:
    # Store current state
    times.append(t)
    states.append(state)
    x, y, vx, vy, theta, omega = state
    
    # Change setpoint halfway through flight
    if t < burn_time / 2:
        desired_angle = initial_setpoint
    else:
        desired_angle = final_setpoint
    
    # PID controller to adjust thrust angle
    error = desired_angle - theta
    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    phi = - (Kp * error + Ki * integral_error + Kd * derivative_error)
    thrust_history.append(phi)
    error_history.append(error)
    previous_error = error
    
    # Calculate forces and torque
    if t < burn_time:
        # During powered flight
        Fx = F_thrust * np.cos(theta + phi)
        Fy = F_thrust * np.sin(theta + phi)  # Gravity handled separately
        torque = -(L/2) * F_thrust * np.sin(phi)
    else:
        # After thrust cutoff
        Fx = 0
        Fy = 0
        torque = 0
    
    # Update derivatives (Euler integration)
    new_state = [
        x + vx * dt,                     # x position
        y + vy * dt,                     # y position
        vx + (Fx/m) * dt,                # x velocity
        vy + (Fy/m - g) * dt,            # y velocity - gravity always affects
        theta + omega * dt,              # angle
        omega + (torque/I) * dt          # angular velocity
    ]
    
    state = np.array(new_state)
    t += dt
    
    # Check if rocket has hit the ground
    if state[1] < 0:
        break

# Convert lists to numpy arrays
times = np.array(times)
states = np.array(states)
thrust_history = np.array(thrust_history)
error_history = np.array(error_history)

# Extract results
x = states[:, 0]
y = states[:, 1]
theta = states[:, 4]

# ======================
# Animation
# ======================
fig, ax = plt.subplots(figsize=(10, 6))
# Set fixed view limits based on trajectory
margin = 100  # increased margin around the trajectory
ax.set_xlim(min(x) - margin, max(x) + margin)
ax.set_ylim(min(min(y) - margin, -margin), max(y) + margin)  # Ensure ground is visible
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Manual Rocket Simulation with PID Controller')
ax.grid(True)
ax.set_aspect('equal')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Add ground line

rocket_line, = ax.plot([], [], 'r-', lw=3)
thrust_vector, = ax.plot([], [], 'y-', lw=2)
trail, = ax.plot([], [], 'b:', alpha=0.5)

def update(frame):
    # Current state
    x_curr = x[frame]
    y_curr = y[frame]
    theta_curr = theta[frame]
    phi_curr = thrust_history[frame]
    t_curr = times[frame]
    
    # Rocket body
    tip = [x_curr + (L/2)*np.cos(theta_curr), 
           y_curr + (L/2)*np.sin(theta_curr)]
    tail = [x_curr - (L/2)*np.cos(theta_curr), 
            y_curr - (L/2)*np.sin(theta_curr)]
    rocket_line.set_data([tail[0], tip[0]], [tail[1], tip[1]])
    
    # Thrust vector (only show when thrust is active)
    if t_curr < burn_time:
        thrust_length = 150
        thrust_vector.set_data(
            [x_curr, x_curr + thrust_length*np.cos(theta_curr + phi_curr)],
            [y_curr, y_curr + thrust_length*np.sin(theta_curr + phi_curr)]
        )
    else:
        thrust_vector.set_data([], [])
    
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