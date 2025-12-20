"""
Stellar Flyby N-Body Simulation
================================
Simulates the gravitational impact of a rogue star passing through a planetary system.

Author: Lisha Ramon
Course: Computational Methods Final Project
Date: December 2025

Physics:
- Newton's law of gravitation: F = G * m1 * m2 / r^2
- Uses leapfrog integration (symplectic, conserves energy well for orbits)
- Units: AU, Solar masses, Years (simplifies G to ~39.478 AU^3 / (M_sun * yr^2))
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple
import json

# =============================================================================
# CONSTANTS (Using AU, Solar masses, Years as units)
# =============================================================================

G = 39.478  # Gravitational constant in AU^3 / (M_sun * yr^2)
# This comes from: G = 4 * pi^2 AU^3 / (M_sun * yr^2)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Body:
    """
    Represents a celestial body in the simulation.
    
    Attributes:
        name: Human-readable identifier
        mass: Mass in solar masses (M_sun)
        position: 3D position vector [x, y, z] in AU
        velocity: 3D velocity vector [vx, vy, vz] in AU/yr
        color: For visualization
        size: Marker size for plotting
    """
    name: str
    mass: float
    position: np.ndarray
    velocity: np.ndarray
    color: str = 'white'
    size: float = 10
    
    def __post_init__(self):
        # Ensure position and velocity are numpy arrays
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)


# =============================================================================
# PHYSICS ENGINE
# =============================================================================

def compute_acceleration(bodies: List[Body], softening: float = 0.01) -> np.ndarray:
    """
    Calculate gravitational acceleration on each body from all other bodies.
    
    The softening parameter prevents numerical singularities when bodies
    get very close (r -> 0 would cause a -> infinity).
    
    Args:
        bodies: List of Body objects
        softening: Softening length in AU (prevents division by zero)
        
    Returns:
        Array of shape (n_bodies, 3) containing acceleration vectors
    """
    n = len(bodies)
    accelerations = np.zeros((n, 3))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # Vector from body i to body j
                r_vec = bodies[j].position - bodies[i].position
                
                # Distance with softening to prevent singularity
                r_mag = np.sqrt(np.sum(r_vec**2) + softening**2)
                
                # Gravitational acceleration: a = G * M / r^2 * r_hat
                accelerations[i] += G * bodies[j].mass * r_vec / r_mag**3
                
    return accelerations


def leapfrog_step(bodies: List[Body], dt: float, softening: float = 0.01) -> None:
    """
    Advance the simulation by one timestep using the Leapfrog integrator.
    
    Leapfrog is a symplectic integrator, meaning it conserves energy well
    over long integration times - crucial for orbital mechanics!
    
    The algorithm:
        1. Kick: v(t + dt/2) = v(t) + a(t) * dt/2
        2. Drift: x(t + dt) = x(t) + v(t + dt/2) * dt
        3. Kick: v(t + dt) = v(t + dt/2) + a(t + dt) * dt/2
    
    Args:
        bodies: List of Body objects (modified in place)
        dt: Timestep in years
        softening: Softening parameter for close encounters
    """
    # First half-kick: update velocities by half timestep
    accelerations = compute_acceleration(bodies, softening)
    for i, body in enumerate(bodies):
        body.velocity += 0.5 * dt * accelerations[i]
    
    # Drift: update positions by full timestep
    for body in bodies:
        body.position += dt * body.velocity
    
    # Second half-kick: update velocities by half timestep with new positions
    accelerations = compute_acceleration(bodies, softening)
    for i, body in enumerate(bodies):
        body.velocity += 0.5 * dt * accelerations[i]


def compute_total_energy(bodies: List[Body]) -> Tuple[float, float, float]:
    """
    Calculate the total mechanical energy of the system.
    
    Energy conservation is a key diagnostic - if energy drifts significantly,
    your timestep is probably too large or there's a bug.
    
    Returns:
        (kinetic_energy, potential_energy, total_energy)
    """
    kinetic = 0.0
    potential = 0.0
    
    for i, body in enumerate(bodies):
        # Kinetic energy: KE = 0.5 * m * v^2
        kinetic += 0.5 * body.mass * np.sum(body.velocity**2)
        
        # Potential energy (count each pair once)
        for j in range(i + 1, len(bodies)):
            r = np.linalg.norm(bodies[j].position - body.position)
            if r > 0:
                potential -= G * body.mass * bodies[j].mass / r
                
    return kinetic, potential, kinetic + potential


def compute_orbital_elements(body: Body, central_mass: float) -> dict:
    """
    Calculate orbital elements for a body orbiting a central mass.
    
    Useful for tracking how the flyby affects each planet's orbit.
    
    Args:
        body: The orbiting body
        central_mass: Mass of the central body (star) in solar masses
        
    Returns:
        Dictionary with semi-major axis (a), eccentricity (e), 
        orbital energy, and angular momentum magnitude
    """
    r = np.linalg.norm(body.position)
    v = np.linalg.norm(body.velocity)
    
    # Specific orbital energy: epsilon = v^2/2 - GM/r
    mu = G * central_mass
    epsilon = 0.5 * v**2 - mu / r
    
    # Semi-major axis: a = -mu / (2 * epsilon)
    # For bound orbits, epsilon < 0, so a > 0
    if epsilon < 0:
        a = -mu / (2 * epsilon)
    else:
        a = float('inf')  # Unbound (hyperbolic) orbit
    
    # Angular momentum vector: L = r x v
    L_vec = np.cross(body.position, body.velocity)
    L_mag = np.linalg.norm(L_vec)
    
    # Eccentricity: e = sqrt(1 + 2*epsilon*L^2 / mu^2)
    if mu > 0:
        e_squared = 1 + 2 * epsilon * L_mag**2 / mu**2
        e = np.sqrt(max(0, e_squared))  # Prevent numerical issues
    else:
        e = 0
    
    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'orbital_energy': epsilon,
        'angular_momentum': L_mag,
        'bound': epsilon < 0
    }


# =============================================================================
# INITIAL CONDITIONS SETUP
# =============================================================================

def create_solar_system() -> List[Body]:
    """
    Create a simplified solar system with the Sun and outer planets.
    
    Initial conditions are approximate circular orbits in the x-y plane.
    Velocities calculated from v = sqrt(G * M_sun / r) for circular orbit.
    
    Returns:
        List of Body objects representing the solar system
    """
    bodies = []
    
    # Sun at the origin
    bodies.append(Body(
        name="Sun",
        mass=1.0,  # 1 solar mass
        position=[0, 0, 0],
        velocity=[0, 0, 0],
        color='yellow',
        size=200
    ))
    
    # Planet data: (name, mass in M_sun, distance in AU, color, size)
    # Mass in Jupiter masses: Jupiter=1, Saturn=0.3, Uranus=0.046, Neptune=0.054
    # Converting to solar masses (Jupiter = 0.000955 M_sun)
    planets = [
        ("Jupiter", 0.000955, 5.2, 'orange', 50),
        ("Saturn", 0.000286, 9.5, 'gold', 40),
        ("Uranus", 0.0000437, 19.2, 'lightblue', 25),
        ("Neptune", 0.0000515, 30.1, 'blue', 25),
    ]
    
    for name, mass, distance, color, size in planets:
        # Circular orbit velocity: v = sqrt(G * M / r)
        v_circular = np.sqrt(G * 1.0 / distance)
        
        bodies.append(Body(
            name=name,
            mass=mass,
            position=[distance, 0, 0],
            velocity=[0, v_circular, 0],  # Perpendicular to radius for circular orbit
            color=color,
            size=size
        ))
    
    return bodies


def create_rogue_star(
    mass: float = 0.5,
    impact_parameter: float = 50.0,
    velocity_inf: float = 20.0,
    approach_angle: float = 0.0,
    inclination: float = 0.0
) -> Body:
    """
    Create a rogue star on a hyperbolic trajectory.
    
    Args:
        mass: Mass in solar masses (typical red dwarf: 0.1-0.5 M_sun)
        impact_parameter: Closest approach distance in AU (if no gravity)
        velocity_inf: Velocity at infinity in AU/yr (~1 AU/yr ≈ 4.74 km/s)
        approach_angle: Angle of approach in x-y plane (radians)
        inclination: Angle above/below the ecliptic (radians)
    
    Returns:
        Body object representing the rogue star
    """
    # Start the rogue star far away (200 AU)
    start_distance = 200.0
    
    # Calculate starting position
    # The star approaches from the +x direction, offset by impact parameter in y
    x_start = start_distance * np.cos(approach_angle)
    y_start = start_distance * np.sin(approach_angle) + impact_parameter
    z_start = start_distance * np.tan(inclination)
    
    # Velocity points roughly toward the origin
    # The star is moving in the -x direction primarily
    vx = -velocity_inf * np.cos(approach_angle)
    vy = -velocity_inf * np.sin(approach_angle)
    vz = -velocity_inf * np.sin(inclination)
    
    return Body(
        name="Rogue Star",
        mass=mass,
        position=[x_start, y_start, z_start],
        velocity=[vx, vy, vz],
        color='red',
        size=150
    )


# =============================================================================
# SIMULATION RUNNER
# =============================================================================

class Simulation:
    """
    Main simulation class that runs the N-body integration and stores results.
    """
    
    def __init__(self, bodies: List[Body], dt: float = 0.01, softening: float = 0.01):
        """
        Initialize the simulation.
        
        Args:
            bodies: List of Body objects
            dt: Timestep in years (0.01 yr = ~3.65 days)
            softening: Softening parameter for gravity calculation
        """
        self.bodies = bodies
        self.dt = dt
        self.softening = softening
        
        # Storage for trajectory history
        self.times = [0.0]
        self.positions = {body.name: [body.position.copy()] for body in bodies}
        self.energies = [compute_total_energy(bodies)]
        self.orbital_elements = {body.name: [] for body in bodies if body.name != "Sun"}
        
    def step(self) -> None:
        """Advance simulation by one timestep."""
        leapfrog_step(self.bodies, self.dt, self.softening)
        
    def run(self, duration: float, save_interval: int = 10, verbose: bool = True) -> None:
        """
        Run the simulation for a given duration.
        
        Args:
            duration: Total time to simulate in years
            save_interval: Save data every N steps (reduces memory usage)
            verbose: Print progress updates
        """
        n_steps = int(duration / self.dt)
        
        for i in range(n_steps):
            self.step()
            
            # Save data at intervals
            if i % save_interval == 0:
                current_time = (i + 1) * self.dt
                self.times.append(current_time)
                
                for body in self.bodies:
                    self.positions[body.name].append(body.position.copy())
                
                self.energies.append(compute_total_energy(self.bodies))
                
                # Track orbital elements for planets
                sun_mass = self.bodies[0].mass
                for body in self.bodies[1:]:
                    elements = compute_orbital_elements(body, sun_mass)
                    self.orbital_elements[body.name].append(elements)
            
            # Progress update
            if verbose and (i + 1) % (n_steps // 10) == 0:
                progress = 100 * (i + 1) / n_steps
                energy = compute_total_energy(self.bodies)[2]
                print(f"Progress: {progress:.0f}% | Time: {(i+1)*self.dt:.1f} yr | Energy: {energy:.6f}")
    
    def get_trajectory_arrays(self) -> dict:
        """Convert position history to numpy arrays for plotting."""
        return {
            name: np.array(positions) 
            for name, positions in self.positions.items()
        }


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_trajectories_3d(sim: Simulation, title: str = "Stellar Flyby Simulation"):
    """
    Create a 3D plot of all body trajectories.
    
    Args:
        sim: Completed Simulation object
        title: Plot title
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    trajectories = sim.get_trajectory_arrays()
    
    for body in sim.bodies:
        traj = trajectories[body.name]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                color=body.color, alpha=0.7, linewidth=1, label=body.name)
        # Mark final position
        ax.scatter(*traj[-1], color=body.color, s=body.size, edgecolor='white')
    
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    
    # Set equal aspect ratio
    max_range = 100  # AU
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    return fig, ax


def plot_orbital_evolution(sim: Simulation):
    """
    Plot how orbital elements change over time.
    
    This is key for showing the impact of the flyby!
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = np.array(sim.times[1:])  # Skip initial time
    
    # Plot for each planet
    for body_name, elements_list in sim.orbital_elements.items():
        if not elements_list:
            continue
            
        # Extract data
        semi_major = [e['semi_major_axis'] for e in elements_list]
        eccentricity = [e['eccentricity'] for e in elements_list]
        
        # Truncate to match times array
        n = min(len(times), len(semi_major))
        
        axes[0, 0].plot(times[:n], semi_major[:n], label=body_name)
        axes[0, 1].plot(times[:n], eccentricity[:n], label=body_name)
    
    # Energy plot
    kinetic = [e[0] for e in sim.energies]
    potential = [e[1] for e in sim.energies]
    total = [e[2] for e in sim.energies]
    
    axes[1, 0].plot(sim.times, kinetic, 'r-', label='Kinetic', alpha=0.7)
    axes[1, 0].plot(sim.times, potential, 'b-', label='Potential', alpha=0.7)
    axes[1, 0].plot(sim.times, total, 'k-', label='Total', linewidth=2)
    
    # Energy conservation (relative error)
    initial_energy = total[0]
    energy_error = [(e - initial_energy) / abs(initial_energy) * 100 for e in total]
    axes[1, 1].plot(sim.times, energy_error, 'k-')
    
    # Labels
    axes[0, 0].set_xlabel('Time (years)')
    axes[0, 0].set_ylabel('Semi-major axis (AU)')
    axes[0, 0].set_title('Orbital Semi-major Axis Evolution')
    axes[0, 0].legend()
    axes[0, 0].set_ylim([0, 100])
    
    axes[0, 1].set_xlabel('Time (years)')
    axes[0, 1].set_ylabel('Eccentricity')
    axes[0, 1].set_title('Orbital Eccentricity Evolution')
    axes[0, 1].legend()
    
    axes[1, 0].set_xlabel('Time (years)')
    axes[1, 0].set_ylabel('Energy (code units)')
    axes[1, 0].set_title('System Energy')
    axes[1, 0].legend()
    
    axes[1, 1].set_xlabel('Time (years)')
    axes[1, 1].set_ylabel('Energy Error (%)')
    axes[1, 1].set_title('Energy Conservation Check')
    
    plt.tight_layout()
    return fig, axes


def create_animation(sim: Simulation, filename: str = 'flyby_animation.gif', 
                     interval: int = 50, trail_length: int = 50):
    """
    Create an animated visualization of the simulation.
    
    Args:
        sim: Completed Simulation object
        filename: Output filename (use .gif or .mp4)
        interval: Milliseconds between frames
        trail_length: Number of past positions to show as trail
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    trajectories = sim.get_trajectory_arrays()
    n_frames = len(sim.times)
    
    # Create line and point objects for each body
    lines = {}
    points = {}
    for body in sim.bodies:
        line, = ax.plot([], [], [], color=body.color, alpha=0.5, linewidth=1)
        point, = ax.plot([], [], [], 'o', color=body.color, markersize=np.sqrt(body.size))
        lines[body.name] = line
        points[body.name] = point
    
    # Set up axes
    max_range = 100
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        for body in sim.bodies:
            lines[body.name].set_data([], [])
            lines[body.name].set_3d_properties([])
            points[body.name].set_data([], [])
            points[body.name].set_3d_properties([])
        time_text.set_text('')
        return list(lines.values()) + list(points.values()) + [time_text]
    
    def animate(frame):
        start = max(0, frame - trail_length)
        
        for body in sim.bodies:
            traj = trajectories[body.name]
            
            # Trail
            lines[body.name].set_data(traj[start:frame+1, 0], traj[start:frame+1, 1])
            lines[body.name].set_3d_properties(traj[start:frame+1, 2])
            
            # Current position
            points[body.name].set_data([traj[frame, 0]], [traj[frame, 1]])
            points[body.name].set_3d_properties([traj[frame, 2]])
        
        time_text.set_text(f'Time: {sim.times[frame]:.1f} years')
        return list(lines.values()) + list(points.values()) + [time_text]
    
    anim = FuncAnimation(fig, animate, init_func=init, frames=n_frames,
                         interval=interval, blit=True)
    
    print(f"Saving animation to {filename}...")
    anim.save(filename, writer='pillow', fps=20)
    print("Done!")
    
    plt.close()
    return anim


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function demonstrating a stellar flyby simulation.
    """
    print("=" * 60)
    print("STELLAR FLYBY SIMULATION")
    print("=" * 60)
    
    # Create the solar system
    print("\nInitializing solar system...")
    solar_system = create_solar_system()
    
    # Create a rogue star
    # Try different parameters to see different outcomes!
    print("Creating rogue star...")
    rogue_star = create_rogue_star(
        mass=0.5,              # Half solar mass (red dwarf)
        impact_parameter=40.0,  # Passes 40 AU from the Sun
        velocity_inf=10.0,      # ~47 km/s relative velocity
        approach_angle=0.0,     # Approaches from +x direction
        inclination=0.1         # Slight inclination above ecliptic
    )
    
    # Combine all bodies
    all_bodies = solar_system + [rogue_star]
    
    # Print initial conditions
    print("\nInitial Conditions:")
    print("-" * 40)
    for body in all_bodies:
        print(f"{body.name}: mass={body.mass:.6f} M_sun, "
              f"pos={body.position}, vel={body.velocity}")
    
    # Create and run simulation
    print("\nRunning simulation...")
    sim = Simulation(all_bodies, dt=0.01, softening=0.01)
    sim.run(duration=200.0, save_interval=10, verbose=True)  # 50 years
    
    # Generate visualizations
    print("\nGenerating plots...")
    
    # 3D trajectory plot
    fig1, ax1 = plot_trajectories_3d(sim, "Stellar Flyby: Rogue Star (0.5 M☉) at 40 AU")
    fig1.savefig('trajectories_3d.png', dpi=150, facecolor='black')
    print("Saved: trajectories_3d.png")
    
    # Orbital evolution plot
    fig2, ax2 = plot_orbital_evolution(sim)
    fig2.savefig('orbital_evolution.png', dpi=150)
    print("Saved: orbital_evolution.png")
    
    # Create animation (optional - takes longer)
    # create_animation(sim, 'flyby_animation.gif')
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    
    # Final orbital state
    print("\nFinal Orbital Elements:")
    print("-" * 40)
    sun_mass = all_bodies[0].mass
    for body in all_bodies[1:]:
        if body.name != "Rogue Star":
            elements = compute_orbital_elements(body, sun_mass)
            status = "BOUND" if elements['bound'] else "EJECTED!"
            print(f"{body.name}: a={elements['semi_major_axis']:.2f} AU, "
                  f"e={elements['eccentricity']:.3f} [{status}]")
    
    plt.show()


if __name__ == "__main__":
    main()
