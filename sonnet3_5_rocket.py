import numpy as np
import matplotlib.pyplot as plt

class RocketState:
    def __init__(self):
        self.x = 0          # position x
        self.y = 0          # position y
        self.vx = 0         # velocity x
        self.vy = 0         # velocity y
        self.theta = 0      # orientation
        self.omega = 0      # angular velocity

class RocketParams:
    def __init__(self):
        self.mass = 10      # kg
        self.length = 2     # m
        self.thrust = 200   # N
        self.burn_time = 8  # s
        self.I = (1/12) * self.mass * self.length**2

def calculate_forces(state, params, phi, t):
    thrust = params.thrust if t <= params.burn_time else 0
    Fx = thrust * np.cos(state.theta + phi)
    Fy = thrust * np.sin(state.theta + phi) - params.mass * 9.81
    torque = (params.length/2) * thrust * np.sin(phi)
    return Fx, Fy, torque

def simulate_rocket(params, total_time=10, dt=0.01, phi=np.deg2rad(-3)):
    state = RocketState()
    times = np.arange(0, total_time, dt)
    states = []
    
    for t in times:
        # Calculate forces
        Fx, Fy, torque = calculate_forces(state, params, phi, t)
        
        # Update linear motion
        ax = Fx / params.mass
        ay = Fy / params.mass
        
        state.vx += ax * dt
        state.vy += ay * dt
        state.x += state.vx * dt
        state.y += state.vy * dt
        
        # Update angular motion
        alpha = torque / params.I
        state.omega += alpha * dt
        state.theta += state.omega * dt
        
        # Store state
        states.append([state.x, state.y, state.theta])
    
    return np.array(states), times

def plot_trajectory(states, times):
    plt.figure(figsize=(10, 6))
    plt.plot(states[:, 0], states[:, 1])
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.title('Rocket Trajectory')
    plt.grid(True)
    plt.axis('equal')
    plt.show()

# Run simulation
params = RocketParams()
states, times = simulate_rocket(params)
plot_trajectory(states, times)