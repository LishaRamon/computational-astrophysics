"""
HW6 - Space Garbage

A heavy steel rod and a spherical ball-bearing, discarded by a passing spaceship, 
are floating in zero gravity and the ball bearing is orbiting around the rod under the effect of its gravitational pull.

For simplicity we will assume that the rod is of negligible cross-section and heavy enough that it doesn’t move significantly, 
and that the ball bearing is orbiting around the rod’s mid-point in a plane perpendicular to the rod.

a) Treating the rod as a line of mass M and length L and the ball bearing as a point of mass m, convince yourself that the 
attractive force F felt by the ball bearing in the direction toward the center of the rod is give by of

F = \frac{GMm}{\sqrt{(x^2 + y^2)(x^2 + y^2 + \frac{1}{4}L^2)}} 

Hence one finds that the equations of motion for the position (x,y) of the ball bearing in the xy-plane are

\frac{d^2 x}{dt^2} = -GM \frac{x}{r^2\sqrt{r^2 + \frac{1}{4}L^2}}  \, \, \, \frac{d^2 y}{dt^2} = -GM \frac{y}{r^2\sqrt{r^2 + \frac{1}{4}L^2}} 

where r = \sqrt{x^2 + y^2} .

b) Convert these two second-order equations into four first-order equations. 
Then working in units of G = 1, write a program to solve for M=10, L=2 and initial conditions (x,y) = (1,0) with a velocity of +1 in the y direction. 
Calculate the orbit from t=0 to t=10 and make a plot of it, meaning a plot of y against x. 
You should find that the ball bearing does not orbit in a circle or an ellipse as a planet does, but has a precessing orbit, which arises 
because the attractive force is not 1/r2 as it is for the Sun.
"""

"""
Space Garbage!
Simulates a ball bearing orbiting a heavy steel rod in zero gravity

Two methods implemented:
1. SciPy's solve_ivp (built-in ODE solver)
2. Manual 4th-order Runge-Kutta (RK4) implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# Physical Parameters (using G = 1 units)

G = 1       # Gravitational constant (normalized)
M = 10      # Mass of the rod
L = 2       # Length of the rod

# initial conditions
x0, y0 = 1, 0       # Starting position: (1, 0)
vx0, vy0 = 0, 1     # Starting velocity: +1 in y direction

# time span
t_start, t_end = 0, 10
dt = 0.001  # time step for RK4 method



# The Equations of Motion

"""
    Computing derivatives for the 4 first-order ODEs
    
    The state vector is [x, y, vx, vy] where:
    - x, y = position of ball bearing
    - vx, vy = velocity components
    
    Returns [dx/dt, dy/dt, dvx/dt, dvy/dt]
    """

def derivatives(t, state):
    
    x, y, vx, vy = state
    
    # distance from origin (center of rod)
    r = np.sqrt(x**2 + y**2)
    
    # common factor in acceleration equations
    # from formula: -GM * (coordinate) / (r^2 * sqrt(r^2 + L^2/4))
    denominator = r**2 * np.sqrt(r**2 + (L**2) / 4)
    
    # accelerations (from the given equations of motion)
    ax = -G * M * x / denominator
    ay = -G * M * y / denominator
    
    # derivatives returned: [dx/dt, dy/dt, dvx/dt, dvy/dt]
    return [vx, vy, ax, ay]


# Method 1: Scipy ODE Solver

"""
    Solving the orbit using SciPy's solve_ivp function
    This is the easier approach utilizing python's built-in tools
"""

def solve_with_scipy():
    
    # initial state vector: [x, y, vx, vy]
    start_state = [x0, y0, vx0, vy0]
    
    # time points of solution location
    t_eval = np.linspace(t_start, t_end, 10000)
    
    # Use the ODE system
    # RK45 is a good default method (adaptive Runge-Kutta)
    solution = solve_ivp(
        derivatives,            # the derivative function
        (t_start, t_end),       # time span
        start_state,          # starting conditions
        method='RK45',          # Runge-Kutta 4(5) method
        t_eval=t_eval,          # where to store solution
        rtol=1e-8               # relative tolerance for accuracy
    )
    
    # find x and y positions from solution
    x_vals = solution.y[0]
    y_vals = solution.y[1]
    
    return x_vals, y_vals



# Method 2: Manual RK4 Implementation

"""
    Performing one step of 4th-order Runge-Kutta integration
    
    RK4 uses 4 slope estimates to get a more accurate result:
    k1 = slope at start
    k2 = slope at midpoint using k1
    k3 = slope at midpoint using k2
    k4 = slope at end using k3
    
    Final estimate = weighted average of all 4 slopes
    """

def rk4_step(state, t, dt):
    
    # convert state to numpy array for vector operations
    state = np.array(state)
    
    # calculate the four slopes
    k1 = np.array(derivatives(t, state))
    k2 = np.array(derivatives(t + dt/2, state + dt/2 * k1))
    k3 = np.array(derivatives(t + dt/2, state + dt/2 * k2))
    k4 = np.array(derivatives(t + dt, state + dt * k3))
    
    # combine slopes with RK4 weights: (k1 + 2*k2 + 2*k3 + k4) / 6
    new_state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
    
    return new_state


"""
    Solving the orbit using manual RK4 implementation
    This shows how the numerical method works step-by-step
"""
def solve_with_rk4():
    
    # Starting state: [x, y, vx, vy]
    state = np.array([x0, y0, vx0, vy0])
    
    # Creating lists to store trajectory
    x_vals = [x0]
    y_vals = [y0]
    
    # num of time steps
    num_steps = int((t_end - t_start) / dt)
    
    # main integration loop
    t = t_start
    for _ in range(num_steps):
        # take one RK4 step
        state = rk4_step(state, t, dt)
        t += dt
        
        # store position
        x_vals.append(state[0])
        y_vals.append(state[1])
    
    return np.array(x_vals), np.array(y_vals)



# Main: Running Both Methods and Plot Results

if __name__ == "__main__":
    print("Solving orbital motion...")
    
    # Solve using both methods
    print("  Method 1: SciPy solve_ivp...")
    x_scipy, y_scipy = solve_with_scipy()
    
    print("  Method 2: Manual RK4...")
    x_rk4, y_rk4 = solve_with_rk4()
    
    print("Finito! Creating plots...")
    
    # show side-by-side comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: SciPy solution
    axes[0].plot(x_scipy, y_scipy, 'b-', linewidth=0.5)
    axes[0].plot(x0, y0, 'go', markersize=8, label='Start')  # Starting point
    axes[0].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[0].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('SciPy solve_ivp Method')
    axes[0].set_aspect('equal')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: RK4 solution
    axes[1].plot(x_rk4, y_rk4, 'r-', linewidth=0.5)
    axes[1].plot(x0, y0, 'go', markersize=8, label='Start')  # Starting point
    axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    axes[1].axvline(x=0, color='gray', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Manual RK4 Method')
    axes[1].set_aspect('equal')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Space Garbage: Ball Bearing Orbiting a Rod\n(Flower pattern: what happens when F ≠ GMm/r² [Rod gravity ≠ Point mass gravity])', 
                 fontsize=12)
    plt.tight_layout()
    
    # saving plot
    plt.savefig('orbit_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved as 'orbit_comparison.png'")
    
    plt.show()
