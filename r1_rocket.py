import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ======================
# Parameters
# ======================
m = 10          # Mass (kg)
L = 6.0         # Length (m) - Increased length
I = (1/12)*m*L**2  # Moment of inertia
g = 9.81        # Gravity (m/s²)
F_thrust = 200  # Thrust force (N)
dt = 0.001      # Smaller time step for quicker controller reaction
burn_time = 4   # Thrust duration
total_time = 30 # Maximum simulation duration

# PID controller parameters
Kp = 5
Ki = 0.001
Kd = 1

# Angle definition: relative to vertical (0°).
# Positive = clockwise from vertical, Negative = counterclockwise.
setpoint = np.deg2rad(20)  # Constant desired angle (45°)

# Nozzle constraints
MAX_NOZZLE_ANGLE = np.deg2rad(5)    # Maximum deflection angle (radians)
MAX_NOZZLE_RATE = np.deg2rad(100)     # Maximum rotation rate (radians/second)
previous_phi = 0                      # Track previous nozzle angle

def limit_nozzle_movement(new_phi, prev_phi, dt):
    """Limit nozzle angle and rotation rate"""
    # Limit rotation rate
    max_delta = MAX_NOZZLE_RATE * dt
    delta_phi = new_phi - prev_phi
    delta_phi = np.clip(delta_phi, -max_delta, max_delta)
    phi = prev_phi + delta_phi
    # Limit absolute angle
    phi = np.clip(phi, -MAX_NOZZLE_ANGLE, MAX_NOZZLE_ANGLE)
    return phi

# Initial state (pointing straight up: theta0 = 0°)
# state vector: [x, y, vx, vy, theta, omega]
state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# ======================
# Manual Euler Integration
# ======================
times = []
states = []
thrust_history = []  # nozzle deflection history (phi)
error_history = []   # PID error history

integral_error = 0.0
previous_error = 0.0
t = 0.0

while t < total_time:
    times.append(t)
    states.append(state)
    x, y, vx, vy, theta, omega = state
    
    # Set desired_angle: 0° setpoint for first 2 seconds, then 20°
    if t < 2:
        desired_angle = np.deg2rad(0)
    else:
        desired_angle = np.deg2rad(20)
    
    # PID controller: error = desired - current angle
    error = desired_angle - theta
    integral_error += error * dt
    derivative_error = (error - previous_error) / dt
    raw_phi = - (Kp * error + Ki * integral_error + Kd * derivative_error)
    
    # Apply nozzle constraints (using previous phi value)
    phi = limit_nozzle_movement(raw_phi, previous_phi, dt)
    previous_phi = phi
    thrust_history.append(phi)
    error_history.append(error)
    previous_error = error
    
    # Calculate forces and torque
    if t < burn_time:
        # Forces computed in new coordinate system:
        # x-force from thrust is F_thrust*sin(theta+phi); y-force is F_thrust*cos(theta+phi)
        Fx = F_thrust * np.sin(theta + phi)
        Fy = F_thrust * np.cos(theta + phi)
        torque = -(L/2) * F_thrust * np.sin(phi)
    else:
        Fx = 0
        Fy = 0
        torque = 0

    # Euler Integration for state update
    new_state = [
        x + vx * dt,                   # x position
        y + vy * dt,                   # y position
        vx + (Fx/m) * dt,              # x velocity
        vy + ((Fy/m) - g) * dt,         # y velocity (gravity always acts)
        theta + omega * dt,            # angle (theta)
        omega + (torque/I) * dt        # angular velocity (omega)
    ]
    
    state = np.array(new_state)
    t += dt
    
    # Stop simulation when rocket hits the ground
    if state[1] < 0:
        break

# Convert lists to numpy arrays
times = np.array(times)
states = np.array(states)
thrust_history = np.array(thrust_history)
error_history = np.array(error_history)
x = states[:, 0]
y = states[:, 1]
theta = states[:, 4]

# ======================
# Animation: Rocket Trajectory
# ======================
fig, ax = plt.subplots(figsize=(10, 6))
margin = 100
ax.set_xlim(min(x) - margin, max(x) + margin)
ax.set_ylim(min(min(y) - margin, -margin), max(y) + margin)
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_title('Manual Rocket Simulation with PID Controller (Trajectory)')
ax.grid(True)
ax.set_aspect('equal')
ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)  # Ground line

rocket_line, = ax.plot([], [], 'r-', lw=3)
thrust_vector, = ax.plot([], [], 'y-', lw=2)
trail, = ax.plot([], [], 'b:', alpha=0.5)

# Add a text box for live simulation info
sim_info = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

def update(frame):
    x_curr = x[frame]
    y_curr = y[frame]
    theta_curr = theta[frame]
    phi_curr = thrust_history[frame]
    t_curr = times[frame]
    
    # Update live simulation info
    engine_status = "ON" if t_curr < burn_time else "OFF"
    # Determine current setpoint based on time
    current_setpoint = 0 if t_curr < 2 else 20
    sim_info.set_text(f"Time: {t_curr:.2f}s\nSetpoint: {current_setpoint}°\nOrientation: {np.rad2deg(theta_curr):.1f}°\nEngine: {engine_status}")
    
    # Compute rocket body endpoints
    tip = [x_curr + (L/2)*np.sin(theta_curr), y_curr + (L/2)*np.cos(theta_curr)]
    tail = [x_curr - (L/2)*np.sin(theta_curr), y_curr - (L/2)*np.cos(theta_curr)]
    rocket_line.set_data([tail[0], tip[0]], [tail[1], tip[1]])
    
    # Thrust vector (display only during powered flight)
    if t_curr < burn_time:
        thrust_length = -30
        thrust_vector.set_data(
            [x_curr, x_curr + thrust_length*np.sin(theta_curr + phi_curr)],
            [y_curr, y_curr + thrust_length*np.cos(theta_curr + phi_curr)]
        )
    else:
        thrust_vector.set_data([], [])
    
    # Trail of trajectory
    trail.set_data(x[:frame], y[:frame])
    
    return rocket_line, thrust_vector, trail, sim_info

ani = FuncAnimation(fig, update, frames=len(times), interval=dt*1000, blit=True)
plt.show()

# ======================
# Informative Post-Simulation Plots
# ======================
# Create a figure with three subplots
fig2, ax2 = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

# In each subplot, add a vertical dashed line at engine shutoff (burn_time)
for axi in ax2:
    axi.axvline(burn_time, color='gray', linestyle='--', label='Engine Shutdown')

# Plot 1: Rocket Orientation and Setpoint vs Time
ax2[0].plot(times, np.rad2deg(theta), label='Rocket Orientation (°)', color='r')
ax2[0].axhline(np.rad2deg(setpoint), color='k', linestyle='--', label='Setpoint (45°)')
ax2[0].set_ylabel('Angle (degrees)')
ax2[0].set_title('Rocket Orientation vs Time')
ax2[0].legend()
ax2[0].grid(True)

# Plot 2: PID Error vs Time
ax2[1].plot(times, np.rad2deg(error_history), label='Error (°)', color='b')
ax2[1].set_ylabel('Error (degrees)')
ax2[1].set_title('PID Error vs Time')
ax2[1].legend()
ax2[1].grid(True)

# Plot 3: Nozzle Deflection (φ) vs Time
ax2[2].plot(times, np.rad2deg(thrust_history), label='Nozzle Deflection (°)', color='m')
ax2[2].set_xlabel('Time (s)')
ax2[2].set_ylabel('Nozzle Angle (degrees)')
ax2[2].set_title('Nozzle Deflection vs Time')
ax2[2].legend()
ax2[2].grid(True)

plt.tight_layout()
plt.show()