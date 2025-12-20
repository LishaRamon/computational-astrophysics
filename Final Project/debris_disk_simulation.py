"""
Debris Disk Stellar Flyby Simulation

Author: Lisha Ramon
Course: Computational Astrophysics Final Project

Simulates a rogue star passing through a Kuiper belt-like debris disk
and visualizes the scattering effects

Produces:
- debris_disk_3d.html: Interactive 3D visualization
- debris_before_after.html: Side-by-side comparison
- debris_comparison.png: Static image for presentations

Usage:
    python debris_disk_simulation.py

You can modify the parameters in the Configuration section below
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import from the main simulation module
from interactive_simulator import (
    create_solar_system, 
    create_rogue_star, 
    create_debris_disk,
    InteractiveSimulation, 
    compute_orbital_elements
)


# Configuration - Modify these parameters to explore different scenarios

# Rogue star parameters
ROGUE_MASS = 0.7              # Solar masses (typical red dwarf: 0.1-0.8)
IMPACT_PARAMETER = 35.0       # AU - where the star passes closest to Sun
VELOCITY = 8.4                # AU/yr (~28 km/s) - slower = more interaction time
INCLINATION_DEG = 5.0         # Degrees above the ecliptic plane

# Debris disk parameters (Kuiper belt analog)
N_PARTICLES = 300             # Number of test particles
DISK_INNER_RADIUS = 30.0      # AU - inner edge of disk
DISK_OUTER_RADIUS = 60.0      # AU - outer edge of disk
DISK_THICKNESS = 4.0          # AU - vertical extent

# Simulation parameters
DURATION = 60.0              # Years to simulate
INCLUDE_INNER_PLANETS = False # Include Mercury, Venus, Earth, Mars?
INCLUDE_OUTER_PLANETS = True  # Include Jupiter, Saturn, Uranus, Neptune?

# Classification thresholds
PERTURBED_ECCENTRICITY = 0.3  # Eccentricity above this = "perturbed"


# Main Simulation Function

def run_debris_simulation():
    """
    Run the debris disk simulation with the configured parameters
    """
    print("=" * 60)
    print("Debris Disk Stellar Flyby Simulation")
    print("=" * 60)
    
    # Print configuration
    print(f"\nConfiguration:")
    print(f"  Rogue star: {ROGUE_MASS} M☉, {IMPACT_PARAMETER} AU, {VELOCITY} AU/yr")
    print(f"  Debris disk: {N_PARTICLES} particles, {DISK_INNER_RADIUS}-{DISK_OUTER_RADIUS} AU")
    print(f"  Duration: {DURATION} years")
    
    # Create solar system
    print("\nCreating solar system...")
    solar_system = create_solar_system(
        include_inner=INCLUDE_INNER_PLANETS, 
        include_outer=INCLUDE_OUTER_PLANETS
    )
    
    # Create rogue star
    print("Creating rogue star...")
    rogue_star = create_rogue_star(
        mass=ROGUE_MASS,
        impact_parameter=IMPACT_PARAMETER,
        velocity_infinity=VELOCITY,
        inclination=np.radians(INCLINATION_DEG)
    )
    
    # Create debris disk
    print(f"Creating debris disk with {N_PARTICLES} particles...")
    debris = create_debris_disk(
        n_particles=N_PARTICLES,
        inner_radius=DISK_INNER_RADIUS,
        outer_radius=DISK_OUTER_RADIUS,
        thickness=DISK_THICKNESS
    )
    
    # Combine all bodies
    all_bodies = solar_system + [rogue_star] + debris
    print(f"Total bodies: {len(all_bodies)}")
    
    # Run simulation
    print("\nRunning simulation...")
    sim = InteractiveSimulation(
        bodies=all_bodies,
        base_dt=0.03,
        integrator='leapfrog',
        adaptive=True
    )
    
    sim.run(duration=DURATION, save_interval=1.0, verbose=True)
    
    return sim, all_bodies, solar_system


def analyze_results(sim, all_bodies, solar_system):
    """
    Analyze the simulation results and classify debris particle fates
    """
    print("\nAnalyzing results...")
    
    trajectories = sim.get_trajectory_arrays()
    sun_mass = solar_system[0].mass
    
    # Storage for analysis
    n_bound = 0
    n_ejected = 0
    n_perturbed = 0
    
    debris_init_x, debris_init_y, debris_init_z = [], [], []
    debris_final_x, debris_final_y, debris_final_z = [], [], []
    debris_colors = []
    
    # Analyze each debris particle
    for body in all_bodies:
        if body.is_test_particle:
            # Initial position
            init_pos = trajectories[body.name][0]
            debris_init_x.append(init_pos[0])
            debris_init_y.append(init_pos[1])
            debris_init_z.append(init_pos[2])
            
            # Final position
            final_pos = body.position
            debris_final_x.append(final_pos[0])
            debris_final_y.append(final_pos[1])
            debris_final_z.append(final_pos[2])
            
            # Compute orbital elements and classify
            elem = compute_orbital_elements(body, sun_mass)
            
            if not elem['bound']:
                n_ejected += 1
                debris_colors.append('red')
            elif elem['eccentricity'] > PERTURBED_ECCENTRICITY:
                n_perturbed += 1
                debris_colors.append('orange')
            else:
                n_bound += 1
                debris_colors.append('cyan')
    
    # Print summary
    print(f"\nDebris Particle Outcomes:")
    print(f"  Stable (e < {PERTURBED_ECCENTRICITY}): {n_bound} ({100*n_bound/N_PARTICLES:.1f}%)")
    print(f"  Perturbed (e > {PERTURBED_ECCENTRICITY}): {n_perturbed} ({100*n_perturbed/N_PARTICLES:.1f}%)")
    print(f"  Ejected: {n_ejected} ({100*n_ejected/N_PARTICLES:.1f}%)")
    
    # Planet outcomes
    print(f"\nPlanet Final States:")
    planet_names = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']
    for body in all_bodies:
        if body.name in planet_names:
            elem = compute_orbital_elements(body, sun_mass)
            status = "BOUND" if elem['bound'] else "EJECTED!"
            print(f"  {body.name:10s}: a={elem['semi_major_axis']:7.2f} AU, e={elem['eccentricity']:.3f} [{status}]")
    
    results = {
        'trajectories': trajectories,
        'debris_init': (debris_init_x, debris_init_y, debris_init_z),
        'debris_final': (debris_final_x, debris_final_y, debris_final_z),
        'debris_colors': debris_colors,
        'counts': (n_bound, n_perturbed, n_ejected)
    }
    
    return results


def create_3d_visualization(sim, all_bodies, results):
    """
    Shoq an interactive 3D visualization of the debris disk scattering.
    """
    print("\nCreating 3D visualization...")
    
    trajectories = results['trajectories']
    debris_init_x, debris_init_y, debris_init_z = results['debris_init']
    debris_final_x, debris_final_y, debris_final_z = results['debris_final']
    debris_colors = results['debris_colors']
    n_bound, n_perturbed, n_ejected = results['counts']
    
    fig = go.Figure()
    
    # Add planet and star trajectories
    for body in all_bodies:
        if not body.is_test_particle:
            traj = trajectories[body.name]
            
            # Trajectory line
            fig.add_trace(go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines',
                name=body.name,
                line=dict(
                    color=body.color, 
                    width=5 if body.name in ['Sun', 'Rogue Star'] else 3
                ),
                opacity=0.9
            ))
            
            # Final position marker
            fig.add_trace(go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode='markers',
                showlegend=False,
                marker=dict(
                    size=12 if body.name in ['Sun', 'Rogue Star'] else 8, 
                    color=body.color
                ),
                hovertemplate=f"<b>{body.name}</b> (final)<extra></extra>"
            ))
    
    # Initial debris disk (ghost ring)
    fig.add_trace(go.Scatter3d(
        x=debris_init_x, y=debris_init_y, z=debris_init_z,
        mode='markers',
        name='Initial debris disk',
        marker=dict(size=2, color='white', opacity=0.3)
    ))
    
    # Final debris positions (colored by fate)
    fig.add_trace(go.Scatter3d(
        x=debris_final_x, y=debris_final_y, z=debris_final_z,
        mode='markers',
        name='Scattered debris',
        marker=dict(size=3, color=debris_colors, opacity=0.8),
        hovertemplate="Debris particle<br>X: %{x:.1f} AU<br>Y: %{y:.1f} AU<extra></extra>"
    ))
    
    # Layout
    fig.update_layout(
        title=dict(
            text=(f"Debris Disk Disruption: {n_ejected} ejected, {n_perturbed} perturbed<br>"
                  f"<sub>{ROGUE_MASS} M☉ star passing at {IMPACT_PARAMETER} AU</sub>"),
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis=dict(title='X (AU)', backgroundcolor='black', gridcolor='gray', range=[-200, 200]),
            yaxis=dict(title='Y (AU)', backgroundcolor='black', gridcolor='gray', range=[-200, 200]),
            zaxis=dict(title='Z (AU)', backgroundcolor='black', gridcolor='gray', range=[-100, 100]),
            bgcolor='black',
            aspectmode='cube'
        ),
        paper_bgcolor='black',
        font=dict(color='white'),
        height=800,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.7)')
    )
    
    fig.write_html("debris_disk_3d.html")
    print("Saved: debris_disk_3d.html")
    
    return fig


def create_before_after_html(sim, all_bodies, results):
    """
    Create interactive before/after comparison visualization.
    """
    print("Creating before/after comparison...")
    
    trajectories = results['trajectories']
    debris_init_x, debris_init_y, debris_init_z = results['debris_init']
    debris_final_x, debris_final_y, debris_final_z = results['debris_final']
    debris_colors = results['debris_colors']
    n_bound, n_perturbed, n_ejected = results['counts']
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "BEFORE: Pristine Debris Disk",
            f"AFTER: {n_ejected} Ejected, {n_perturbed} Perturbed"
        )
    )
    
    # Planet orbit circles for reference
    theta = np.linspace(0, 2*np.pi, 100)
    for dist in [30, 19, 9.5, 5.2]:
        for col in [1, 2]:
            fig.add_trace(go.Scatter(
                x=dist*np.cos(theta), y=dist*np.sin(theta),
                mode='lines',
                line=dict(color='rgba(100,100,100,0.5)', width=1, dash='dot'),
                showlegend=False
            ), row=1, col=col)
    
    # Sun
    for col in [1, 2]:
        fig.add_trace(go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=20, color='yellow', line=dict(color='orange', width=2)),
            name='Sun',
            showlegend=(col == 1)
        ), row=1, col=col)
    
    # Before: initial debris
    fig.add_trace(go.Scatter(
        x=debris_init_x, y=debris_init_y,
        mode='markers',
        marker=dict(size=6, color='cyan', opacity=0.7),
        name='Debris particles'
    ), row=1, col=1)
    
    # After: scattered debris (colored by fate)
    fig.add_trace(go.Scatter(
        x=debris_final_x, y=debris_final_y,
        mode='markers',
        marker=dict(size=6, color=debris_colors, opacity=0.8),
        name='Scattered debris'
    ), row=1, col=2)
    
    # Rogue star path
    rogue_traj = trajectories['Rogue Star']
    fig.add_trace(go.Scatter(
        x=rogue_traj[:, 0], y=rogue_traj[:, 1],
        mode='lines',
        line=dict(color='red', width=4),
        name='Rogue Star Path'
    ), row=1, col=2)
    
    # Layout
    fig.update_layout(
        title=dict(
            text="Kuiper Belt Disruption by Stellar Flyby",
            font=dict(size=18, color='white')
        ),
        paper_bgcolor='#0a0a1a',
        plot_bgcolor='#0a0a1a',
        font=dict(color='white'),
        height=600,
        legend=dict(bgcolor='rgba(0,0,0,0.5)')
    )
    
    fig.update_xaxes(range=[-150, 150], gridcolor='rgba(50,50,50,0.5)', zeroline=False)
    fig.update_yaxes(range=[-150, 150], gridcolor='rgba(50,50,50,0.5)', zeroline=False, scaleanchor='x')
    
    fig.write_html("debris_before_after.html")
    print("Saved: debris_before_after.html")
    
    return fig


def create_static_image(sim, all_bodies, results):
    """
    Create static PNG image for presentations.
    """
    print("Creating static comparison image...")
    
    trajectories = results['trajectories']
    debris_init_x, debris_init_y, debris_init_z = results['debris_init']
    debris_final_x, debris_final_y, debris_final_z = results['debris_final']
    debris_colors = results['debris_colors']
    n_bound, n_perturbed, n_ejected = results['counts']
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Helper function to set up axes
    def setup_axis(ax, title):
        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_xlabel('X (AU)', fontsize=12, color='white')
        ax.set_ylabel('Y (AU)', fontsize=12, color='white')
        ax.set_title(title, fontsize=14, color='white', fontweight='bold')
        ax.set_facecolor('#0a0a1a')
        ax.tick_params(colors='white')
        ax.set_aspect('equal')
        for spine in ax.spines.values():
            spine.set_color('#333')
        # Planet orbits
        for dist in [5.2, 9.5, 19, 30]:
            circle = plt.Circle((0, 0), dist, fill=False, color='gray', 
                               linestyle=':', alpha=0.4, linewidth=1)
            ax.add_patch(circle)
    
    # Before panel
    setup_axis(axes[0], 'BEFORE: Pristine Debris Disk')
    axes[0].scatter(debris_init_x, debris_init_y, c='cyan', s=15, alpha=0.7, edgecolors='none')
    axes[0].scatter([0], [0], c='yellow', s=400, marker='*', zorder=10, 
                   edgecolors='orange', linewidths=2)
    
    # After panel
    setup_axis(axes[1], f'AFTER: {n_ejected} Ejected, {n_perturbed} Perturbed')
    axes[1].scatter(debris_final_x, debris_final_y, c=debris_colors, s=15, alpha=0.8, edgecolors='none')
    axes[1].scatter([0], [0], c='yellow', s=400, marker='*', zorder=10, 
                   edgecolors='orange', linewidths=2)
    
    # Rogue star path
    rogue_traj = trajectories['Rogue Star']
    axes[1].plot(rogue_traj[:, 0], rogue_traj[:, 1], 'r-', linewidth=3, 
                label='Rogue Star Path', zorder=5)
    
    # Legend for colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', 
               markersize=10, label=f'Stable ({n_bound})', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
               markersize=10, label=f'Perturbed ({n_perturbed})', linestyle='None'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label=f'Ejected ({n_ejected})', linestyle='None'),
        Line2D([0], [0], color='red', linewidth=3, label='Rogue Star Path'),
    ]
    axes[1].legend(handles=legend_elements, loc='upper right', 
                  facecolor='#1a1a2e', labelcolor='white', fontsize=10)
    
    # Overall figure styling
    fig.patch.set_facecolor('#0a0a1a')
    velocity_kms = VELOCITY * 4.74  # Convert AU/yr to km/s
    plt.suptitle(
        f'Stellar Flyby: Kuiper Belt Disruption\n'
        f'{ROGUE_MASS} M☉ star passing through disk at {IMPACT_PARAMETER} AU, ~{velocity_kms:.0f} km/s',
        fontsize=16, color='white', fontweight='bold', y=0.98
    )
    
    plt.tight_layout()
    fig.savefig('debris_comparison.png', dpi=200, facecolor='#0a0a1a', bbox_inches='tight')
    print("Saved: debris_comparison.png")
    
    return fig


# Main Entry Point

def main():
    """
    Main function to run the complete debris disk simulation and visualization
    """
    # Run simulation
    sim, all_bodies, solar_system = run_debris_simulation()
    
    # Analyze results
    results = analyze_results(sim, all_bodies, solar_system)
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    
    create_3d_visualization(sim, all_bodies, results)
    create_before_after_html(sim, all_bodies, results)
    create_static_image(sim, all_bodies, results)
    
    print("\n" + "=" * 60)
    print("Simulation Complete!!")
    print("=" * 60)
    print("\nOutput files:")
    print("  - debris_disk_3d.html      : Interactive 3D view (open in browser)")
    print("  - debris_before_after.html : Before/after comparison (open in browser)")
    print("  - debris_comparison.png    : Static image for presentations")
    
    return sim, results


if __name__ == "__main__":
    main()
