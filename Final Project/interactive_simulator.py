"""
Interactive Stellar Flyby Simulator
Author: Lisha Ramon
Computational Astrophysics Final Project

A complete simulation tool with user-configurable parameters

Features:
- Full solar system including Earth and inner planets
- User control over rogue star: mass, velocity, impact parameter
- User control over planet masses
- Adaptive timestep during close encounters
- Multiple integration schemes (Leapfrog, RK4)
- Debris disk visualization option
- Interactive parameter input


Simulation Runtime (all planets + 100yrs + 0.1): ~1 hour 20min
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Constants

G = 39.478  # Gravitational constant in AU^3 / (M_sun * yr^2)

# Conversion factors for user convenience
KM_S_TO_AU_YR = 0.210945  # 1 km/s ≈ 0.211 AU/yr
AU_YR_TO_KM_S = 4.74047   # 1 AU/yr ≈ 4.74 km/s



# Data Structures

@dataclass
class Body:
    """
    Represents a celestial body with position, velocity, and physical properties.
    """
    name: str
    mass: float                    # Solar masses
    position: np.ndarray           # AU
    velocity: np.ndarray           # AU/yr
    color: str = 'white'
    size: float = 10               # Marker size for plotting
    is_test_particle: bool = False # Test particles don't exert gravity
    
    def __post_init__(self):
        self.position = np.array(self.position, dtype=float)
        self.velocity = np.array(self.velocity, dtype=float)
        # Store initial conditions for reset
        self._initial_position = self.position.copy()
        self._initial_velocity = self.velocity.copy()
    
    def reset(self):
        """Reset to initial conditions."""
        self.position = self._initial_position.copy()
        self.velocity = self._initial_velocity.copy()



# Solar system and rogue star setup functions


# Default planet data: (name, mass in M_sun, semi-major axis in AU, color, size)
PLANET_DATA = {
    'Mercury': (1.66e-7, 0.387, 'gray', 8),
    'Venus':   (2.45e-6, 0.723, 'orange', 12),
    'Earth':   (3.00e-6, 1.000, 'dodgerblue', 15),
    'Mars':    (3.23e-7, 1.524, 'red', 10),
    'Jupiter': (9.55e-4, 5.203, 'orange', 40),
    'Saturn':  (2.86e-4, 9.537, 'gold', 35),
    'Uranus':  (4.37e-5, 19.19, 'lightblue', 25),
    'Neptune': (5.15e-5, 30.07, 'blue', 25),
}


def create_solar_system(
    include_inner: bool = True,
    include_outer: bool = True,
    planet_mass_multipliers: Dict[str, float] = None
) -> List[Body]:
    """
    Create the solar system with configurable planet masses.
    
    Args:
        include_inner: Include Mercury, Venus, Earth, Mars
        include_outer: Include Jupiter, Saturn, Uranus, Neptune
        planet_mass_multipliers: Dict mapping planet names to mass multipliers
                                 e.g., {'Earth': 2.0} makes Earth twice as massive
    
    Returns:
        List of Body objects (Sun + selected planets)
    """
    if planet_mass_multipliers is None:
        planet_mass_multipliers = {}
    
    bodies = []
    
    # Sun at origin (will be adjusted for center of mass later)
    bodies.append(Body(
        name="Sun",
        mass=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        color='yellow',
        size=100
    ))
    
    inner_planets = ['Mercury', 'Venus', 'Earth', 'Mars']
    outer_planets = ['Jupiter', 'Saturn', 'Uranus', 'Neptune']
    
    planets_to_add = []
    if include_inner:
        planets_to_add.extend(inner_planets)
    if include_outer:
        planets_to_add.extend(outer_planets)
    
    for planet_name in planets_to_add:
        base_mass, distance, color, size = PLANET_DATA[planet_name]
        
        # Apply mass multiplier if specified
        multiplier = planet_mass_multipliers.get(planet_name, 1.0)
        mass = base_mass * multiplier
        
        # Circular orbit velocity
        v_circular = np.sqrt(G * 1.0 / distance)
        
        bodies.append(Body(
            name=planet_name,
            mass=mass,
            position=[distance, 0.0, 0.0],
            velocity=[0.0, v_circular, 0.0],
            color=color,
            size=size
        ))
    
    return bodies


def create_rogue_star(
    mass: float = 0.5,
    impact_parameter: float = 50.0,
    velocity_infinity: float = 10.0,  # AU/yr
    approach_angle: float = 0.0,      # radians in x-y plane
    inclination: float = 0.0,         # radians above ecliptic
    start_distance: float = 200.0     # AU
) -> Body:
    """
    Create a rogue star on a hyperbolic trajectory.
    
    Args:
        mass: Mass in solar masses
        impact_parameter: Perpendicular distance from Sun if no gravity (AU)
        velocity_infinity: Speed at infinity in AU/yr (multiply by 4.74 for km/s)
        approach_angle: Direction of approach in ecliptic plane
        inclination: Angle above/below ecliptic
        start_distance: Starting distance from Sun
    
    Returns:
        Body object for the rogue star
    """
    # Position: start far away, offset by impact parameter
    x = start_distance * np.cos(approach_angle)
    y = start_distance * np.sin(approach_angle) + impact_parameter * np.cos(inclination)
    z = impact_parameter * np.sin(inclination)
    
    # Velocity: pointing roughly toward the Sun
    vx = -velocity_infinity * np.cos(approach_angle) * np.cos(inclination)
    vy = -velocity_infinity * np.sin(approach_angle)
    vz = -velocity_infinity * np.sin(inclination)
    
    return Body(
        name="Rogue Star",
        mass=mass,
        position=[x, y, z],
        velocity=[vx, vy, vz],
        color='red',
        size=80
    )


def create_debris_disk(
    n_particles: int = 500,
    inner_radius: float = 30.0,
    outer_radius: float = 50.0,
    thickness: float = 5.0
) -> List[Body]:
    """
    Create a debris disk of test particles (e.g., Kuiper belt analog).
    
    Test particles feel gravity but don't exert it, making the simulation
    much faster while still showing how debris gets scattered.
    
    Args:
        n_particles: Number of test particles
        inner_radius: Inner edge of disk (AU)
        outer_radius: Outer edge of disk (AU)
        thickness: Vertical thickness (AU)
    
    Returns:
        List of test particle Body objects
    """
    particles = []
    
    for i in range(n_particles):
        # Random position in disk
        r = np.random.uniform(inner_radius, outer_radius)
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(-thickness/2, thickness/2)
        
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # Circular orbit velocity (around Sun)
        v_circ = np.sqrt(G * 1.0 / r)
        vx = -v_circ * np.sin(theta)
        vy = v_circ * np.cos(theta)
        vz = 0.0
        
        particles.append(Body(
            name=f"particle_{i}",
            mass=0.0,  # Massless
            position=[x, y, z],
            velocity=[vx, vy, vz],
            color='white',
            size=1,
            is_test_particle=True
        ))
    
    return particles



# Physics enginer with Leapfrog and RK4 integrator

def compute_acceleration(
    bodies: List[Body], 
    softening: float = 0.001
) -> np.ndarray:
    """
    Compute gravitational acceleration on each body.
    
    Test particles don't contribute to the gravitational field but do
    feel accelerations from massive bodies.
    """
    n = len(bodies)
    accelerations = np.zeros((n, 3))
    
    for i in range(n):
        for j in range(n):
            if i != j and bodies[j].mass > 0:  # Only massive bodies exert gravity
                r_vec = bodies[j].position - bodies[i].position
                r_mag = np.sqrt(np.sum(r_vec**2) + softening**2)
                accelerations[i] += G * bodies[j].mass * r_vec / r_mag**3
    
    return accelerations


def leapfrog_step(bodies: List[Body], dt: float, softening: float = 0.001) -> None:
    """
    Leapfrog integrator: symplectic, excellent energy conservation.
    
    Best for: Long-term orbital evolution
    """
    # Half kick
    acc = compute_acceleration(bodies, softening)
    for i, body in enumerate(bodies):
        body.velocity += 0.5 * dt * acc[i]
    
    # Full drift
    for body in bodies:
        body.position += dt * body.velocity
    
    # Half kick
    acc = compute_acceleration(bodies, softening)
    for i, body in enumerate(bodies):
        body.velocity += 0.5 * dt * acc[i]


def rk4_step(bodies: List[Body], dt: float, softening: float = 0.001) -> None:
    """
    4th-order Runge-Kutta integrator: high accuracy, handles rapid changes.
    
    Best for: Close encounters with rapidly changing accelerations
    """
    n = len(bodies)
    
    # Store initial state
    pos0 = np.array([b.position.copy() for b in bodies])
    vel0 = np.array([b.velocity.copy() for b in bodies])
    
    # k1
    acc1 = compute_acceleration(bodies, softening)
    k1_v = acc1
    k1_x = vel0
    
    # k2: advance to midpoint using k1
    for i, body in enumerate(bodies):
        body.position = pos0[i] + 0.5 * dt * k1_x[i]
        body.velocity = vel0[i] + 0.5 * dt * k1_v[i]
    acc2 = compute_acceleration(bodies, softening)
    k2_v = acc2
    k2_x = np.array([b.velocity for b in bodies])
    
    # k3: advance to midpoint using k2
    for i, body in enumerate(bodies):
        body.position = pos0[i] + 0.5 * dt * k2_x[i]
        body.velocity = vel0[i] + 0.5 * dt * k2_v[i]
    acc3 = compute_acceleration(bodies, softening)
    k3_v = acc3
    k3_x = np.array([b.velocity for b in bodies])
    
    # k4: advance to end using k3
    for i, body in enumerate(bodies):
        body.position = pos0[i] + dt * k3_x[i]
        body.velocity = vel0[i] + dt * k3_v[i]
    acc4 = compute_acceleration(bodies, softening)
    k4_v = acc4
    k4_x = np.array([b.velocity for b in bodies])
    
    # Combine: y_new = y_old + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
    for i, body in enumerate(bodies):
        body.position = pos0[i] + (dt/6) * (k1_x[i] + 2*k2_x[i] + 2*k3_x[i] + k4_x[i])
        body.velocity = vel0[i] + (dt/6) * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])


def compute_min_distance(bodies: List[Body]) -> float:
    """Find minimum distance between any two massive bodies."""
    min_dist = float('inf')
    for i, b1 in enumerate(bodies):
        if b1.mass == 0:
            continue
        for j, b2 in enumerate(bodies):
            if i < j and b2.mass > 0:
                dist = np.linalg.norm(b1.position - b2.position)
                min_dist = min(min_dist, dist)
    return min_dist


def adaptive_timestep(
    bodies: List[Body],
    base_dt: float,
    min_dt: float = 0.0001,
    distance_threshold: float = 10.0
) -> float:
    """
    Compute adaptive timestep based on closest approach.
    
    When bodies are close together, we need smaller timesteps to
    accurately capture the rapid changes in acceleration.
    
    Args:
        bodies: Current body list
        base_dt: Default timestep when bodies are far apart
        min_dt: Minimum allowed timestep
        distance_threshold: Distance below which we start reducing dt
    
    Returns:
        Adjusted timestep
    """
    min_dist = compute_min_distance(bodies)
    
    if min_dist < distance_threshold:
        # Scale timestep with distance: closer = smaller dt
        # Using a smooth scaling function
        scale = max(min_dist / distance_threshold, min_dt / base_dt)
        return base_dt * scale
    
    return base_dt



# Analysis Functions

def compute_total_energy(bodies: List[Body]) -> Tuple[float, float, float]:
    """Calculate kinetic, potential, and total energy."""
    kinetic = 0.0
    potential = 0.0
    
    for i, body in enumerate(bodies):
        if body.mass > 0:
            kinetic += 0.5 * body.mass * np.sum(body.velocity**2)
        
        for j in range(i + 1, len(bodies)):
            if bodies[j].mass > 0 and body.mass > 0:
                r = np.linalg.norm(bodies[j].position - body.position)
                if r > 0:
                    potential -= G * body.mass * bodies[j].mass / r
    
    return kinetic, potential, kinetic + potential


def compute_orbital_elements(body: Body, central_mass: float) -> dict:
    """Compute orbital elements relative to the central mass."""
    r = np.linalg.norm(body.position)
    v = np.linalg.norm(body.velocity)
    
    mu = G * central_mass
    epsilon = 0.5 * v**2 - mu / r  # Specific orbital energy
    
    # Semi-major axis
    if epsilon < 0:
        a = -mu / (2 * epsilon)
    else:
        a = float('inf')
    
    # Angular momentum
    L_vec = np.cross(body.position, body.velocity)
    L_mag = np.linalg.norm(L_vec)
    
    # Eccentricity
    if mu > 0:
        e_sq = 1 + 2 * epsilon * L_mag**2 / mu**2
        e = np.sqrt(max(0, e_sq))
    else:
        e = 0
    
    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'orbital_energy': epsilon,
        'angular_momentum': L_mag,
        'bound': epsilon < 0
    }



# Simulation class

class InteractiveSimulation:
    """
    Main simulation class with support for adaptive timestep and
    multiple integration methods.
    """
    
    def __init__(
        self,
        bodies: List[Body],
        base_dt: float = 0.01,
        integrator: str = 'leapfrog',  # 'leapfrog' or 'rk4'
        adaptive: bool = True,
        softening: float = 0.001
    ):
        """
        Initialize simulation.
        
        Args:
            bodies: List of Body objects
            base_dt: Base timestep in years
            integrator: Integration method ('leapfrog' or 'rk4')
            adaptive: Whether to use adaptive timestep
            softening: Gravitational softening parameter
        """
        self.bodies = bodies
        self.base_dt = base_dt
        self.integrator = integrator
        self.adaptive = adaptive
        self.softening = softening
        
        # Select integration function
        self.step_func = leapfrog_step if integrator == 'leapfrog' else rk4_step
        
        # Data storage
        self.times = [0.0]
        self.positions = {b.name: [b.position.copy()] for b in bodies}
        self.energies = [compute_total_energy(bodies)]
        self.timesteps = [base_dt]  # Track adaptive timestep values
        
    def step(self) -> float:
        """
        Advance simulation by one timestep.
        
        Returns:
            The timestep used
        """
        if self.adaptive:
            dt = adaptive_timestep(self.bodies, self.base_dt)
        else:
            dt = self.base_dt
        
        self.step_func(self.bodies, dt, self.softening)
        return dt
    
    def run(
        self,
        duration: float,
        save_interval: float = 0.1,  # Save every 0.1 years
        verbose: bool = True
    ) -> None:
        """
        Run simulation for specified duration.
        
        Args:
            duration: Total time in years
            save_interval: Time between saved snapshots
            verbose: Print progress
        """
        current_time = 0.0
        last_save = 0.0
        step_count = 0
        
        while current_time < duration:
            dt = self.step()
            current_time += dt
            step_count += 1
            
            # Save data at intervals
            if current_time - last_save >= save_interval:
                self.times.append(current_time)
                for body in self.bodies:
                    self.positions[body.name].append(body.position.copy())
                self.energies.append(compute_total_energy(self.bodies))
                self.timesteps.append(dt)
                last_save = current_time
            
            # Progress update
            if verbose and step_count % 1000 == 0:
                progress = 100 * current_time / duration
                print(f"  {progress:5.1f}% | t={current_time:.2f} yr | dt={dt:.6f} yr | steps={step_count}")
        
        if verbose:
            print(f"  Complete: {step_count} total steps")
    
    def get_trajectory_arrays(self) -> Dict[str, np.ndarray]:
        """Convert position history to numpy arrays."""
        return {name: np.array(pos) for name, pos in self.positions.items()}
    
    def get_final_orbital_elements(self) -> Dict[str, dict]:
        """Get final orbital elements for all planets."""
        sun_mass = self.bodies[0].mass
        elements = {}
        
        for body in self.bodies:
            if body.name not in ["Sun", "Rogue Star"] and body.mass > 0:
                elements[body.name] = compute_orbital_elements(body, sun_mass)
        
        return elements



# Interactive user inputs


def get_user_parameters() -> dict:
    """
    Interactively get simulation parameters from the user.
    
    Returns:
        Dictionary of all user-specified parameters
    """
    print("\n" + "=" * 60)
    print("STELLAR FLYBY SIMULATOR - Configuration")
    print("=" * 60)
    
    params = {}
    
    # Rogue star parameters
    print("\n--- ROGUE STAR PARAMETERS ---")
    print("(Press Enter for default values)\n")
    
    val = input("Rogue star mass (solar masses) [0.5]: ").strip()
    params['rogue_mass'] = float(val) if val else 0.5
    
    val = input("Impact parameter (AU) [50]: ").strip()
    params['impact_parameter'] = float(val) if val else 50.0
    
    val = input("Velocity at infinity (km/s) [50]: ").strip()
    v_kms = float(val) if val else 50.0
    params['velocity'] = v_kms * KM_S_TO_AU_YR  # Convert to AU/yr
    params['velocity_kms'] = v_kms
    
    val = input("Inclination above ecliptic (degrees) [0]: ").strip()
    params['inclination'] = np.radians(float(val) if val else 0.0)
    
    # Planet selection
    print("\n--- PLANET SELECTION ---")
    val = input("Include inner planets (Mercury-Mars)? [y]/n: ").strip().lower()
    params['include_inner'] = val != 'n'
    
    val = input("Include outer planets (Jupiter-Neptune)? [y]/n: ").strip().lower()
    params['include_outer'] = val != 'n'
    
    # Custom planet masses
    print("\n--- CUSTOM PLANET MASSES (optional) ---")
    print("Enter mass multipliers (e.g., '2.0' makes planet twice as massive)")
    print("Press Enter to keep default mass\n")
    
    params['planet_masses'] = {}
    
    if params['include_inner']:
        for planet in ['Mercury', 'Venus', 'Earth', 'Mars']:
            val = input(f"  {planet} mass multiplier [1.0]: ").strip()
            if val:
                params['planet_masses'][planet] = float(val)
    
    if params['include_outer']:
        for planet in ['Jupiter', 'Saturn', 'Uranus', 'Neptune']:
            val = input(f"  {planet} mass multiplier [1.0]: ").strip()
            if val:
                params['planet_masses'][planet] = float(val)
    
    # Debris disk
    print("\n--- DEBRIS DISK (optional) ---")
    val = input("Add debris disk (Kuiper belt)? y/[n]: ").strip().lower()
    params['add_debris'] = val == 'y'
    
    if params['add_debris']:
        val = input("  Number of particles [200]: ").strip()
        params['n_debris'] = int(val) if val else 200
    
    # Simulation settings
    print("\n--- SIMULATION SETTINGS ---")
    
    val = input("Simulation duration (years) [100]: ").strip()
    params['duration'] = float(val) if val else 100.0
    
    val = input("Integrator (leapfrog/rk4) [leapfrog]: ").strip().lower()
    params['integrator'] = val if val in ['leapfrog', 'rk4'] else 'leapfrog'
    
    val = input("Use adaptive timestep? [y]/n: ").strip().lower()
    params['adaptive'] = val != 'n'
    
    val = input("Base timestep (years) [0.01]: ").strip()
    params['base_dt'] = float(val) if val else 0.01
    
    return params


def print_parameter_summary(params: dict) -> None:
    """Print a summary of the simulation parameters."""
    print("\n" + "=" * 60)
    print("SIMULATION CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print(f"\nRogue Star:")
    print(f"  Mass: {params['rogue_mass']:.3f} M☉")
    print(f"  Impact parameter: {params['impact_parameter']:.1f} AU")
    print(f"  Velocity: {params['velocity_kms']:.1f} km/s ({params['velocity']:.2f} AU/yr)")
    print(f"  Inclination: {np.degrees(params['inclination']):.1f}°")
    
    print(f"\nPlanets:")
    print(f"  Inner planets: {'Yes' if params['include_inner'] else 'No'}")
    print(f"  Outer planets: {'Yes' if params['include_outer'] else 'No'}")
    if params['planet_masses']:
        print(f"  Mass modifications:")
        for planet, mult in params['planet_masses'].items():
            print(f"    {planet}: ×{mult}")
    
    if params['add_debris']:
        print(f"\nDebris Disk: {params['n_debris']} particles")
    
    print(f"\nSimulation:")
    print(f"  Duration: {params['duration']:.1f} years")
    print(f"  Integrator: {params['integrator'].upper()}")
    print(f"  Adaptive timestep: {'Yes' if params['adaptive'] else 'No'}")
    print(f"  Base dt: {params['base_dt']} years")



# Viszualization 


def create_comprehensive_visualization(
    sim: InteractiveSimulation,
    params: dict,
    show_timestep: bool = True
) -> go.Figure:
    """
    Create a comprehensive interactive visualization with multiple panels.
    
    Includes: 3D trajectories, orbital evolution, timestep history
    """
    trajectories = sim.get_trajectory_arrays()
    times = np.array(sim.times)
    
    # Determine subplot layout based on options
    if show_timestep:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "scatter3d", "rowspan": 2}, {"type": "scatter"}],
                [None, {"type": "scatter"}]
            ],
            subplot_titles=(
                "3D Trajectories",
                "Orbital Evolution",
                "Adaptive Timestep"
            ),
            column_widths=[0.6, 0.4],
            vertical_spacing=0.12
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scatter3d"}, {"type": "scatter"}]],
            subplot_titles=("3D Trajectories", "Orbital Evolution"),
            column_widths=[0.6, 0.4]
        )
    
    # 3D Trajectories
    for body in sim.bodies:
        if body.is_test_particle:
            continue  # Skip individual particles in legend
        
        traj = trajectories[body.name]
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines',
                name=body.name,
                line=dict(color=body.color, width=4 if body.name in ["Sun", "Rogue Star"] else 2),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Final position marker
        fig.add_trace(
            go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode='markers',
                name=f"{body.name} (final)",
                marker=dict(size=8, color=body.color),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add debris disk if present (as a scatter cloud)
    debris_final_x = []
    debris_final_y = []
    debris_final_z = []
    for body in sim.bodies:
        if body.is_test_particle:
            debris_final_x.append(body.position[0])
            debris_final_y.append(body.position[1])
            debris_final_z.append(body.position[2])
    
    if debris_final_x:
        fig.add_trace(
            go.Scatter3d(
                x=debris_final_x, y=debris_final_y, z=debris_final_z,
                mode='markers',
                name='Debris',
                marker=dict(size=1, color='white', opacity=0.3),
            ),
            row=1, col=1
        )
    
    # Orbital elements over time
    final_elements = sim.get_final_orbital_elements()
    colors = {
        'Mercury': 'gray', 'Venus': 'orange', 'Earth': 'dodgerblue',
        'Mars': 'red', 'Jupiter': 'orange', 'Saturn': 'gold',
        'Uranus': 'lightblue', 'Neptune': 'blue'
    }
    
    # For orbital evolution, we'd need to track elements over time
    # Here we just show final eccentricities as a bar chart
    planet_names = list(final_elements.keys())
    eccentricities = [final_elements[p]['eccentricity'] for p in planet_names]
    bound_status = ['Bound' if final_elements[p]['bound'] else 'EJECTED' for p in planet_names]
    
    fig.add_trace(
        go.Bar(
            x=planet_names,
            y=eccentricities,
            marker_color=[colors.get(p, 'white') for p in planet_names],
            text=bound_status,
            textposition='outside',
            name='Final Eccentricity'
        ),
        row=1, col=2
    )
    
    # Timestep evolution
    if show_timestep and len(sim.timesteps) > 1:
        fig.add_trace(
            go.Scatter(
                x=times[:len(sim.timesteps)],
                y=sim.timesteps,
                mode='lines',
                name='Timestep',
                line=dict(color='cyan', width=2)
            ),
            row=2, col=2
        )
    
    # Layout
    title = (
        f"Stellar Flyby: {params['rogue_mass']:.2f} M☉ at {params['impact_parameter']:.0f} AU, "
        f"v={params['velocity_kms']:.0f} km/s"
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='white')),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        height=700,
        showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.5)')
    )
    
    # 3D scene settings
    max_range = max(100, params['impact_parameter'] * 2)
    fig.update_scenes(
        xaxis=dict(title='X (AU)', backgroundcolor='#1a1a2e', gridcolor='gray',
                   range=[-max_range, max_range]),
        yaxis=dict(title='Y (AU)', backgroundcolor='#1a1a2e', gridcolor='gray',
                   range=[-max_range, max_range]),
        zaxis=dict(title='Z (AU)', backgroundcolor='#1a1a2e', gridcolor='gray',
                   range=[-max_range/2, max_range/2]),
        bgcolor='#1a1a2e',
        aspectmode='cube'
    )
    
    # 2D axes
    fig.update_xaxes(title_text="Planet", gridcolor='gray', row=1, col=2)
    fig.update_yaxes(title_text="Eccentricity", gridcolor='gray', row=1, col=2)
    
    if show_timestep:
        fig.update_xaxes(title_text="Time (years)", gridcolor='gray', row=2, col=2)
        fig.update_yaxes(title_text="Timestep (years)", gridcolor='gray', type='log', row=2, col=2)
    
    return fig


def create_timestep_analysis_plot(sim: InteractiveSimulation) -> plt.Figure:
    """
    Create a detailed plot showing how the adaptive timestep responds
    to the flyby encounter.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    times = np.array(sim.times[:len(sim.timesteps)])
    timesteps = np.array(sim.timesteps)
    
    # Find rogue star distance over time
    rogue_traj = sim.get_trajectory_arrays().get('Rogue Star')
    sun_traj = sim.get_trajectory_arrays().get('Sun')
    
    if rogue_traj is not None and sun_traj is not None:
        distances = np.linalg.norm(rogue_traj - sun_traj, axis=1)
        
        # Top panel: Rogue star distance
        axes[0].plot(sim.times[:len(distances)], distances, 'r-', linewidth=2, label='Rogue Star Distance')
        axes[0].axhline(y=50, color='yellow', linestyle='--', alpha=0.5, label='50 AU reference')
        axes[0].set_ylabel('Distance from Sun (AU)', fontsize=12)
        axes[0].set_title('Rogue Star Approach and Timestep Response', fontsize=14)
        axes[0].legend(loc='upper right')
        axes[0].set_ylim(bottom=0)
        axes[0].grid(True, alpha=0.3)
    
    # Bottom panel: Timestep
    axes[1].semilogy(times, timesteps, 'c-', linewidth=2)
    axes[1].set_xlabel('Time (years)', fontsize=12)
    axes[1].set_ylabel('Timestep (years)', fontsize=12)
    axes[1].set_title('Adaptive Timestep (log scale)', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Highlight the close encounter period
    if rogue_traj is not None:
        min_dist_idx = np.argmin(distances)
        encounter_time = sim.times[min_dist_idx]
        for ax in axes:
            ax.axvline(x=encounter_time, color='red', linestyle=':', alpha=0.7, label='Closest approach')
    
    plt.tight_layout()
    return fig


# main function to run simulation

def run_interactive_simulation(use_defaults: bool = False):
    """
    Main function to run the interactive simulation.
    
    Args:
        use_defaults: If True, skip user input and use default parameters
    """
    if use_defaults:
        # Default parameters for quick testing
        params = {
            'rogue_mass': 0.5,
            'impact_parameter': 40.0,
            'velocity': 50.0 * KM_S_TO_AU_YR,
            'velocity_kms': 50.0,
            'inclination': np.radians(10.0),
            'include_inner': True,
            'include_outer': True,
            'planet_masses': {},
            'add_debris': False,
            'duration': 80.0,
            'integrator': 'leapfrog',
            'adaptive': True,
            'base_dt': 0.01
        }
    else:
        params = get_user_parameters()
    
    print_parameter_summary(params)
    
    # Confirm before running
    if not use_defaults:
        val = input("\nProceed with simulation? [y]/n: ").strip().lower()
        if val == 'n':
            print("Simulation cancelled.")
            return None, None
    
    print("\n" + "=" * 60)
    print("RUNNING SIMULATION")
    print("=" * 60 + "\n")
    
    # Create bodies
    solar_system = create_solar_system(
        include_inner=params['include_inner'],
        include_outer=params['include_outer'],
        planet_mass_multipliers=params['planet_masses']
    )
    
    rogue_star = create_rogue_star(
        mass=params['rogue_mass'],
        impact_parameter=params['impact_parameter'],
        velocity_infinity=params['velocity'],
        inclination=params['inclination']
    )
    
    all_bodies = solar_system + [rogue_star]
    
    # Add debris if requested
    if params.get('add_debris', False):
        debris = create_debris_disk(n_particles=params.get('n_debris', 200))
        all_bodies.extend(debris)
        print(f"Added {len(debris)} debris particles")
    
    # Create and run simulation
    sim = InteractiveSimulation(
        bodies=all_bodies,
        base_dt=params['base_dt'],
        integrator=params['integrator'],
        adaptive=params['adaptive']
    )
    
    sim.run(duration=params['duration'], save_interval=0.1, verbose=True)
    
    # Print final state
    print("\n" + "=" * 60)
    print("FINAL ORBITAL STATE")
    print("=" * 60)
    
    final_elements = sim.get_final_orbital_elements()
    for planet, elem in final_elements.items():
        status = "BOUND" if elem['bound'] else "EJECTED!"
        print(f"  {planet:10s}: a={elem['semi_major_axis']:8.2f} AU, e={elem['eccentricity']:.3f} [{status}]")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Interactive Plotly figure
    fig = create_comprehensive_visualization(sim, params, show_timestep=params['adaptive'])
    fig.write_html("simulation_results.html")
    print("Saved: simulation_results.html")
    
    # Timestep analysis (if adaptive)
    if params['adaptive']:
        fig_ts = create_timestep_analysis_plot(sim)
        fig_ts.savefig("timestep_analysis.png", dpi=150, facecolor='#1a1a2e')
        print("Saved: timestep_analysis.png")
    
    print("\n✓ Simulation complete! Open simulation_results.html in your browser.")
    
    return sim, params



# Entry point

if __name__ == "__main__":
    import sys
    
    # Check for command line argument to use defaults
    use_defaults = '--defaults' in sys.argv or '-d' in sys.argv
    
    if use_defaults:
        print("Running with default parameters...")
    
    sim, params = run_interactive_simulation(use_defaults=use_defaults)
