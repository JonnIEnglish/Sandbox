import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation

# Rocket parameters
m0 = 1000.0       # Initial mass [kg]
Iy = 5000.0       # Pitch moment of inertia [kg·m²]
S = 1.2           # Reference area [m²]
d = 0.3           # Reference length [m]
g = 9.81          # Gravity [m/s²]
rho = 1.225       # Air density [kg/m³]
Isp = 250         # Specific impulse [s]
Ae = 0.2          # Nozzle exit area [m²]

# Aerodynamic coefficients
C_A = 0.1         # Axial force coefficient
C_N = 0.5         # Normal force coefficient
C_m = -0.02       # Pitch moment coefficient
C_mq = -0.1       # Pitch damping derivative

def rocket_dynamics(t, state, delta_e, F_pref):
    """ 2.5D longitudinal dynamics with thrust """
    x, y, u, w, theta, q = state  # Added x, y positions
    V_m = np.sqrt(u**2 + w**2)
    
    # Thrust calculation (Equation 4 simplified)
    F_p = F_pref  # Simplified thrust model (constant for now)
    
    # Aerodynamic forces
    F_Ax = -0.5 * rho * V_m**2 * C_A * S
    F_Az = 0.5 * rho * V_m**2 * C_N * S
    
    # Aerodynamic moment (Equation 12)
    M_A = 0.5 * rho * V_m**2 * d * S * (C_m + C_mq * (q * d)/(2 * V_m) + delta_e)
    
    # Mass depletion (Equation 10 simplified)
    m = m0 - (F_pref/(Isp * g)) * t  # Simple mass model
    
    # Gravity components
    F_gx = -m * g * np.sin(theta)
    F_gz = m * g * np.cos(theta)
    
    # State derivatives with position
    x_dot = u * np.cos(theta) + w * np.sin(theta)
    y_dot = -u * np.sin(theta) + w * np.cos(theta)
    u_dot = (F_Ax + F_p + F_gx)/m - q * w
    w_dot = (F_Az + F_gz)/m + q * u
    theta_dot = q
    q_dot = M_A / Iy
    
    return [x_dot, y_dot, u_dot, w_dot, theta_dot, q_dot]

def simulate_rocket(delta_e=0.0, F_pref=5000.0, t_span=[0, 10]):
    # Initial state: [x (m), y (m), u (m/s), w (m/s), theta (rad), q (rad/s)]
    initial_state = [0.0, 0.0, 20.0, 0.0, 0.0, 0.0]
    
    # Solve ODE
    sol = solve_ivp(lambda t, y: rocket_dynamics(t, y, delta_e, F_pref),
                    t_span, initial_state, 
                    t_eval=np.linspace(t_span[0], t_span[1], 100))
    
    # Plot results
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(sol.y[0]))
    ax.set_ylim(0, max(sol.y[1]))
    rocket, = ax.plot([], [], 'ro')

    def init():
        rocket.set_data([], [])
        return rocket,

    def update(frame):
        rocket.set_data(sol.y[0][frame], sol.y[1][frame])
        return rocket,

    ani = FuncAnimation(fig, update, frames=len(sol.t), init_func=init, blit=True)
    plt.show()

# Run simulation with 5° elevator and 5000N thrust
simulate_rocket(delta_e=np.deg2rad(5), F_pref=5000)