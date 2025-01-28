import matplotlib.pyplot as plt
import numpy as np

mass = 10 # kg
thrust = 200 # N
burn_time = 8 # s
gravity = 9.81 # m/s^2
nozzle_angle = -3 # degrees


def calculate_x_force(thrust, nozzle_angle):
    # Shift angle by 90 degrees so 0 is upward
    return thrust * np.sin(np.deg2rad(nozzle_angle))

def calculate_y_force(thrust, nozzle_angle):
    # Shift angle by 90 degrees so 0 is upward
    return thrust * np.cos(np.deg2rad(nozzle_angle))

def update_y_acceleration(y_force, mass, timestamp, burn_time):
    if timestamp <= burn_time:
        return (y_force / mass) - gravity
    else:
        return -gravity
    
def update_x_acceleration(x_force, mass):
    return x_force / mass

def update_y_velocity(acceleration, time, initial_velocity=0):
    return initial_velocity + acceleration * time

def update_x_velocity(acceleration, time, initial_velocity=0):
    return initial_velocity + acceleration * time

def update_y_position(acceleration, time, initial_velocity=0, initial_position=0):
    return initial_position + initial_velocity * time + 0.5 * acceleration * time * time

def update_x_position(acceleration, time, initial_velocity=0, initial_position=0):
    return initial_position + initial_velocity * time + 0.5 * acceleration * time * time

def get_oscillating_angle(t, amplitude=2, period=2):
    """Returns an oscillating angle between -amplitude and +amplitude degrees"""
    return amplitude * np.sin(2 * np.pi * t / period)

# Calculate acceleration, velocity, and position over time
def simulate_rocket(mass, thrust, initial_nozzle_angle, burn_time, total_time, time_step):
    timestamps = np.arange(0, total_time + time_step, time_step)
    y_accelerations = []
    x_accelerations = []
    y_velocities = []
    x_velocities = []
    y_positions = []
    x_positions = []
    actual_timestamps = []
    
    velocity_y = 0
    position_y = 0
    velocity_x = 0
    position_x = 0
    
    for t in timestamps:
        # Store current timestamp
        actual_timestamps.append(t)

        # Calculate oscillating nozzle angle
        nozzle_angle = get_oscillating_angle(t)
        
        # Calculate thrust components for current timestep
        x_force = calculate_x_force(thrust, nozzle_angle)
        y_force = calculate_y_force(thrust, nozzle_angle)
        
        if t > burn_time:
            thrust = 0
        
        # Calculate accelerations
        accel_y = update_y_acceleration(y_force, mass, t, burn_time)
        accel_x = update_x_acceleration(x_force, mass)
        
        # Update velocities using previous timestep
        velocity_y += accel_y * time_step
        velocity_x += accel_x * time_step
        
        # Update positions using updated velocities
        position_y += velocity_y * time_step
        position_x += velocity_x * time_step
        
        # Store values
        y_accelerations.append(accel_y)
        x_accelerations.append(accel_x)
        y_velocities.append(velocity_y)
        x_velocities.append(velocity_x)
        y_positions.append(position_y)
        x_positions.append(position_x)

        # Check if rocket has hit the ground
        if position_y < 0:
            # Adjust final position to be exactly at ground level
            y_positions[-1] = 0
            break
    
    # Convert to numpy arrays and trim to actual length
    actual_timestamps = np.array(actual_timestamps)
    y_accelerations = np.array(y_accelerations)
    y_velocities = np.array(y_velocities)
    y_positions = np.array(y_positions)
    x_positions = np.array(x_positions)
    
    return actual_timestamps, y_accelerations, y_velocities, y_positions, x_positions

# Example usage:
total_time = 30 # s
time_step = 0.1 # s

timestamps, y_accelerations, y_velocities, y_positions, x_positions = simulate_rocket(mass, thrust, nozzle_angle, burn_time, total_time, time_step)

# Plotting the results
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(timestamps, y_accelerations, label='Y Acceleration')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.title('Vertical Acceleration vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(timestamps, y_velocities, label='Y Velocity', color='orange')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Vertical Velocity vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(timestamps, y_positions, label='Y Position', color='green')
plt.xlabel('Time (s)')
plt.ylabel('Altitude (m)')
plt.title('Altitude vs Time')
plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_positions, y_positions, label='Trajectory', color='red')
plt.xlabel('X Position (m)')
plt.ylabel('Altitude (m)')
plt.title('2D Trajectory')
plt.grid(True)
plt.legend()
plt.axis('equal')  # Add this line to ensure equal scaling
plt.tight_layout()

plt.show()
