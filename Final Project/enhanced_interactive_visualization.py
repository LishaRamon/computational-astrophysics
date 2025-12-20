"""
Expanded Interactive 3D Visualization for Stellar Flyby Simulation
Author: Lisha Ramon
Course: Computational Astrophysics Final Project

Integrates with interactive_simulator.py to visualize results with full features

Features:
- Full solar system (all 8 planets: Mercury through Neptune)
- Toggle debris disk visualization on/off
- Animated debris disk scattering with time slider
- Hover over trajectories to see body names and positions
- Zoom, pan, and rotate the view
- Reflects user-inputted simulation duration
- Export to HTML for presentations
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple

# Import from your interactive_simulator
from interactive_simulator import (
    InteractiveSimulation, Body, create_solar_system, create_rogue_star,
    create_debris_disk, compute_orbital_elements, G, KM_S_TO_AU_YR
)



# Color Schemes and Constants

# Planet colors for consistent visualization
PLANET_COLORS = {
    'Sun': 'yellow',
    'Mercury': '#8c8c8c',      # Gray
    'Venus': '#e6c87a',        # Pale yellow
    'Earth': '#4a90d9',        # Blue
    'Mars': '#d94f30',         # Red-orange
    'Jupiter': '#d9a066',      # Orange-brown
    'Saturn': '#e6d066',       # Gold
    'Uranus': '#7acfd9',       # Light cyan
    'Neptune': '#3366cc',      # Deep blue
    'Rogue Star': '#ff3333'    # Bright red
}

# Planet sizes for markers (relative scaling)
PLANET_SIZES = {
    'Sun': 20,
    'Mercury': 6,
    'Venus': 8,
    'Earth': 8,
    'Mars': 7,
    'Jupiter': 16,
    'Saturn': 14,
    'Uranus': 11,
    'Neptune': 11,
    'Rogue Star': 18
}



# Debris Disk Classification

def classify_debris_particles(
    debris_bodies: List[Body],
    sun_mass: float = 1.0
) -> Tuple[List[str], Dict[str, int]]:
    """
    Classify debris particles by their orbital fate
    
    Returns colors for each particle and a count summary
    
    Categories:
    - Stable (cyan): Eccentricity < 0.3, relatively unaffected
    - Perturbed (orange): Eccentricity 0.3-1.0, significantly altered
    - Ejected (red): Unbound, leaving the system
    """
    colors = []
    counts = {'stable': 0, 'perturbed': 0, 'ejected': 0}
    
    for body in debris_bodies:
        elem = compute_orbital_elements(body, sun_mass)
        
        if not elem['bound']:
            colors.append('red')
            counts['ejected'] += 1
        elif elem['eccentricity'] > 0.3:
            colors.append('orange')
            counts['perturbed'] += 1
        else:
            colors.append('cyan')
            counts['stable'] += 1
    
    return colors, counts



# Static 3D visualization

def create_interactive_3d_plot(
    sim: InteractiveSimulation,
    params: dict,
    show_debris: bool = True,
    title: str = None
) -> go.Figure:
    """
    Below is an interactive 3D plot of all trajectories using Plotly.
    
    You can rotate, zoom, and hover over trajectories in your browser!
    
    Args:
        sim: Completed InteractiveSimulation object
        params: Dictionary of simulation parameters
        show_debris: Whether to show debris disk particles
        title: Plot title (auto-generated if None)
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    trajectories = sim.get_trajectory_arrays()
    
    # Auto-generate title from parameters
    if title is None:
        title = (
            f"Stellar Flyby: {params['rogue_mass']:.2f} M☉ at "
            f"{params['impact_parameter']:.0f} AU, "
            f"v={params['velocity_kms']:.0f} km/s "
            f"({params['duration']:.0f} years)"
        )
    
    # Separate bodies into planets and debris
    planet_bodies = [b for b in sim.bodies if not b.is_test_particle]
    debris_bodies = [b for b in sim.bodies if b.is_test_particle]
    
    # Add trajectory line for each planet/star
    for body in planet_bodies:
        traj = trajectories[body.name]
        color = PLANET_COLORS.get(body.name, body.color)
        
        # The trajectory line
        fig.add_trace(go.Scatter3d(
            x=traj[:, 0],
            y=traj[:, 1],
            z=traj[:, 2],
            mode='lines',
            name=f"{body.name} orbit",
            line=dict(
                color=color,
                width=4 if body.name in ["Sun", "Rogue Star"] else 2
            ),
            opacity=0.7,
            hovertemplate=(
                f"<b>{body.name}</b><br>"
                "X: %{x:.2f} AU<br>"
                "Y: %{y:.2f} AU<br>"
                "Z: %{z:.2f} AU<br>"
                "<extra></extra>"
            )
        ))
        
        # Starting position marker (diamond)
        fig.add_trace(go.Scatter3d(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            z=[traj[0, 2]],
            mode='markers',
            name=f"{body.name} (start)",
            marker=dict(
                size=6,
                color=color,
                symbol='diamond',
                line=dict(color='white', width=1)
            ),
            showlegend=False,
            hovertemplate=f"<b>{body.name} START</b><extra></extra>"
        ))
        
        # Final position marker (larger circle)
        marker_size = PLANET_SIZES.get(body.name, 10)
        fig.add_trace(go.Scatter3d(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            z=[traj[-1, 2]],
            mode='markers',
            name=f"{body.name} (end)",
            marker=dict(
                size=marker_size,
                color=color,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            showlegend=False,
            hovertemplate=f"<b>{body.name} FINAL</b><extra></extra>"
        ))
    
    # Add debris disk if present and requested
    if show_debris and debris_bodies:
        # Initial debris positions
        debris_init_x = [trajectories[b.name][0, 0] for b in debris_bodies]
        debris_init_y = [trajectories[b.name][0, 1] for b in debris_bodies]
        debris_init_z = [trajectories[b.name][0, 2] for b in debris_bodies]
        
        # Final debris positions
        debris_final_x = [b.position[0] for b in debris_bodies]
        debris_final_y = [b.position[1] for b in debris_bodies]
        debris_final_z = [b.position[2] for b in debris_bodies]
        
        # Classify debris by orbital fate
        debris_colors, debris_counts = classify_debris_particles(debris_bodies)
        
        # Initial debris (gray, semi-transparent)
        fig.add_trace(go.Scatter3d(
            x=debris_init_x,
            y=debris_init_y,
            z=debris_init_z,
            mode='markers',
            name='Debris (initial)',
            marker=dict(size=2, color='gray', opacity=0.3),
            hovertemplate="<b>Initial debris</b><extra></extra>"
        ))
        
        # Final debris (colored by fate)
        fig.add_trace(go.Scatter3d(
            x=debris_final_x,
            y=debris_final_y,
            z=debris_final_z,
            mode='markers',
            name='Debris (final)',
            marker=dict(size=3, color=debris_colors, opacity=0.7),
            hovertemplate=(
                "<b>Debris particle</b><br>"
                "X: %{x:.1f} AU<br>"
                "Y: %{y:.1f} AU<br>"
                "Z: %{z:.1f} AU<br>"
                "<extra></extra>"
            )
        ))
        
        # Add annotation for debris statistics
        stats_text = (
            f"Debris: {debris_counts['stable']} stable, "
            f"{debris_counts['perturbed']} perturbed, "
            f"{debris_counts['ejected']} ejected"
        )
        title = f"{title}<br><sub>{stats_text}</sub>"
    
    # Calculate axis ranges based on trajectories
    all_positions = np.concatenate([traj for traj in trajectories.values()])
    x_range = [all_positions[:, 0].min() - 20, all_positions[:, 0].max() + 20]
    y_range = [all_positions[:, 1].min() - 20, all_positions[:, 1].max() + 20]
    z_range = [all_positions[:, 2].min() - 10, all_positions[:, 2].max() + 10]
    
    # Make ranges symmetric for better viewing
    max_xy = max(abs(x_range[0]), abs(x_range[1]), abs(y_range[0]), abs(y_range[1]))
    max_z = max(abs(z_range[0]), abs(z_range[1]))
    
    # Configure the layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title='X (AU)',
                backgroundcolor='black',
                gridcolor='gray',
                showbackground=True,
                range=[-max_xy, max_xy]
            ),
            yaxis=dict(
                title='Y (AU)',
                backgroundcolor='black',
                gridcolor='gray',
                showbackground=True,
                range=[-max_xy, max_xy]
            ),
            zaxis=dict(
                title='Z (AU)',
                backgroundcolor='black',
                gridcolor='gray',
                showbackground=True,
                range=[-max_z, max_z]
            ),
            bgcolor='black',
            aspectmode='cube'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0.7)'
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        height=700
    )
    
    return fig



# Animated 3D visualization with debris

def create_animated_3d_plot(
    sim: InteractiveSimulation,
    params: dict,
    show_debris: bool = True,
    frame_step: int = 5,
    trail_length: int = 50,
    title: str = None
) -> go.Figure:
    """
    Produces an animated 3D plot with time slider showing debris scattering

    You can scrub through the flyby and watch the debris disk scatter in real-time!

    Args:
        sim: Completed InteractiveSimulation object
        params: Dictionary of simulation parameters
        show_debris: Whether to animate debris particles
        frame_step: Use every Nth frame (reduces file size)
        trail_length: Number of points in trajectory trail
        title: Plot title (auto-generated if None)
        
    Returns:
        Plotly figure object with animation
    """
    trajectories = sim.get_trajectory_arrays()
    times = np.array(sim.times)
    n_frames = len(times)
    
    # Auto-generate title
    if title is None:
        title = (
            f"Stellar Flyby Animation: {params['rogue_mass']:.2f} M☉ at "
            f"{params['impact_parameter']:.0f} AU"
        )
    
    # Separate planets and debris
    planet_bodies = [b for b in sim.bodies if not b.is_test_particle]
    debris_bodies = [b for b in sim.bodies if b.is_test_particle]
    
    # Calculate axis ranges
    all_positions = np.concatenate([traj for traj in trajectories.values()])
    max_xy = max(
        abs(all_positions[:, 0].min()), abs(all_positions[:, 0].max()),
        abs(all_positions[:, 1].min()), abs(all_positions[:, 1].max())
    ) + 20
    max_z = max(abs(all_positions[:, 2].min()), abs(all_positions[:, 2].max())) + 10
    
    # Create frames for animation
    frames = []
    
    for i in range(0, n_frames, frame_step):
        frame_data = []
        
        # Add planet trajectories and current positions
        for body in planet_bodies:
            traj = trajectories[body.name]
            color = PLANET_COLORS.get(body.name, body.color)
            
            # Trail (last N points up to current time)
            trail_start = max(0, i - trail_length)
            
            # Trajectory trail
            frame_data.append(go.Scatter3d(
                x=traj[trail_start:i+1, 0],
                y=traj[trail_start:i+1, 1],
                z=traj[trail_start:i+1, 2],
                mode='lines',
                line=dict(color=color, width=3),
                opacity=0.6,
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # Current position marker
            marker_size = PLANET_SIZES.get(body.name, 10)
            frame_data.append(go.Scatter3d(
                x=[traj[i, 0]],
                y=[traj[i, 1]],
                z=[traj[i, 2]],
                mode='markers',
                marker=dict(
                    size=marker_size,
                    color=color,
                    line=dict(color='white', width=2)
                ),
                name=body.name,
                showlegend=(i == 0),
                hovertemplate=f"<b>{body.name}</b><br>t={times[i]:.1f} yr<extra></extra>"
            ))
        
        # Add debris particles if enabled
        if show_debris and debris_bodies:
            debris_x = [trajectories[b.name][i, 0] for b in debris_bodies]
            debris_y = [trajectories[b.name][i, 1] for b in debris_bodies]
            debris_z = [trajectories[b.name][i, 2] for b in debris_bodies]
            
            # Color debris by current state
            # For animation, we use a simpler approach: color by distance from origin
            debris_dist = np.sqrt(
                np.array(debris_x)**2 + 
                np.array(debris_y)**2 + 
                np.array(debris_z)**2
            )
            
            # Create color scale: close = cyan, far = orange, very far = red
            debris_colors = []
            for d in debris_dist:
                if d > 100:
                    debris_colors.append('red')      # Likely ejected
                elif d > 60:
                    debris_colors.append('orange')   # Perturbed
                else:
                    debris_colors.append('cyan')     # Relatively stable
            
            frame_data.append(go.Scatter3d(
                x=debris_x,
                y=debris_y,
                z=debris_z,
                mode='markers',
                marker=dict(size=2, color=debris_colors, opacity=0.6),
                name='Debris',
                showlegend=(i == 0),
                hoverinfo='skip'
            ))
        
        frames.append(go.Frame(
            data=frame_data,
            name=f"t={times[i]:.1f}",
            layout=go.Layout(
                title=dict(
                    text=f"{title}<br><sub>Time: {times[i]:.1f} / {params['duration']:.0f} years</sub>",
                    font=dict(size=16, color='white')
                )
            )
        ))
    
    # Create initial figure with first frame's data
    fig = go.Figure(data=frames[0].data, frames=frames)
    
    # Add play/pause buttons and time slider
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>Time: 0.0 / {params['duration']:.0f} years</sub>",
            font=dict(size=16, color='white')
        ),
        scene=dict(
            xaxis=dict(
                title='X (AU)',
                backgroundcolor='black',
                gridcolor='gray',
                range=[-max_xy, max_xy]
            ),
            yaxis=dict(
                title='Y (AU)',
                backgroundcolor='black',
                gridcolor='gray',
                range=[-max_xy, max_xy]
            ),
            zaxis=dict(
                title='Z (AU)',
                backgroundcolor='black',
                gridcolor='gray',
                range=[-max_z, max_z]
            ),
            bgcolor='black',
            aspectmode='cube'
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        height=700,
        margin=dict(l=0, r=0, t=80, b=80),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0,
                x=0.1,
                xanchor="right",
                yanchor="top",
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": 80, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0}
                        }]
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }]
                    )
                ]
            )
        ],
        sliders=[{
            "active": 0,
            "yanchor": "top",
            "xanchor": "left",
            "currentvalue": {
                "font": {"size": 14, "color": "white"},
                "prefix": "Time: ",
                "suffix": " years",
                "visible": True,
                "xanchor": "center"
            },
            "transition": {"duration": 0},
            "pad": {"b": 10, "t": 50},
            "len": 0.9,
            "x": 0.05,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }],
                    "label": f"{times[i]:.0f}",
                    "method": "animate"
                }
                for i, f in zip(range(0, n_frames, frame_step), frames)
            ]
        }]
    )
    
    return fig



# Debris disk before/after comparison

def create_debris_comparison_plot(
    sim: InteractiveSimulation,
    params: dict
) -> go.Figure:
    """
    Have a side-by-side comparison showing debris disk before and after flyby
    
    Shows initial pristine disk vs scattered debris with color coding
    """
    trajectories = sim.get_trajectory_arrays()
    debris_bodies = [b for b in sim.bodies if b.is_test_particle]
    
    if not debris_bodies:
        print("Warning: No debris particles found in simulation")
        return None
    
    # Get initial and final positions
    debris_init_x = [trajectories[b.name][0, 0] for b in debris_bodies]
    debris_init_y = [trajectories[b.name][0, 1] for b in debris_bodies]
    
    debris_final_x = [b.position[0] for b in debris_bodies]
    debris_final_y = [b.position[1] for b in debris_bodies]
    
    # Classify debris
    debris_colors, debris_counts = classify_debris_particles(debris_bodies)
    
    # Create side-by-side plot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "BEFORE: Initial Debris Disk",
            f"AFTER: Scattered ({debris_counts['stable']} stable, "
            f"{debris_counts['perturbed']} perturbed, "
            f"{debris_counts['ejected']} ejected)"
        ),
        horizontal_spacing=0.1
    )
    
    # Before panel: initial positions
    fig.add_trace(go.Scatter(
        x=debris_init_x,
        y=debris_init_y,
        mode='markers',
        marker=dict(size=4, color='cyan', opacity=0.6),
        name='Initial'
    ), row=1, col=1)
    
    # Add planet orbit circles for reference
    theta = np.linspace(0, 2*np.pi, 100)
    orbit_data = [
        ('Neptune', 30.07, 'blue'),
        ('Uranus', 19.19, 'lightblue'),
        ('Saturn', 9.537, 'gold'),
        ('Jupiter', 5.203, 'orange'),
        ('Mars', 1.524, 'red'),
        ('Earth', 1.0, 'dodgerblue')
    ]
    
    for name, dist, color in orbit_data:
        if params.get('include_outer', True) or dist < 2:
            fig.add_trace(go.Scatter(
                x=dist * np.cos(theta),
                y=dist * np.sin(theta),
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                name=name,
                showlegend=False
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=dist * np.cos(theta),
                y=dist * np.sin(theta),
                mode='lines',
                line=dict(color=color, width=1, dash='dash'),
                showlegend=False
            ), row=1, col=2)
    
    # After panel: final positions with fate colors
    fig.add_trace(go.Scatter(
        x=debris_final_x,
        y=debris_final_y,
        mode='markers',
        marker=dict(size=4, color=debris_colors, opacity=0.7),
        name='Final'
    ), row=1, col=2)
    
    # Sun markers
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=15, color='yellow', symbol='star'),
        name='Sun',
        showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0], y=[0],
        mode='markers',
        marker=dict(size=15, color='yellow', symbol='star'),
        showlegend=False
    ), row=1, col=2)
    
    # Layout
    title = (
        f"Debris Disk Disruption: {params['rogue_mass']:.2f} M☉ at "
        f"{params['impact_parameter']:.0f} AU"
    )
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color='white')),
        paper_bgcolor='#1a1a2e',
        plot_bgcolor='#1a1a2e',
        font=dict(color='white'),
        height=600,
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
    )
    
    # Set axis ranges
    max_range = max(80, params.get('impact_parameter', 50) * 2)
    
    for col in [1, 2]:
        fig.update_xaxes(
            title_text="X (AU)",
            gridcolor='gray',
            range=[-max_range, max_range],
            row=1, col=col
        )
        fig.update_yaxes(
            title_text="Y (AU)",
            gridcolor='gray',
            range=[-max_range, max_range],
            row=1, col=col
        )
    
    return fig



# Comprehensive Dashboard

def create_comprehensive_dashboard(
    sim: InteractiveSimulation,
    params: dict,
    show_debris: bool = True,
    show_timestep: bool = True
) -> go.Figure:
    """
    Below is a comprehensive multi-panel dashboard showing all simulation results
    
    Panels include:
    - 3D trajectories (all planets + debris)
    - Final eccentricity bar chart
    - Orbital semi-major axis evolution (if tracked)
    - Timestep evolution (if adaptive)
    """
    trajectories = sim.get_trajectory_arrays()
    times = np.array(sim.times)
    final_elements = sim.get_final_orbital_elements()
    
    # Determine layout based on options
    n_cols = 2
    n_rows = 2 if show_timestep else 1
    
    specs = [
        [{"type": "scatter3d", "rowspan": n_rows}, {"type": "bar"}],
    ]
    if show_timestep:
        specs.append([None, {"type": "scatter"}])
    
    subplot_titles = ["3D Trajectories", "Final Orbital Eccentricity"]
    if show_timestep:
        subplot_titles.append("Adaptive Timestep")
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        column_widths=[0.6, 0.4],
        vertical_spacing=0.15
    )
    
    # Panel 1: 3D Trajectories
    planet_bodies = [b for b in sim.bodies if not b.is_test_particle]
    debris_bodies = [b for b in sim.bodies if b.is_test_particle]
    
    for body in planet_bodies:
        traj = trajectories[body.name]
        color = PLANET_COLORS.get(body.name, body.color)
        
        fig.add_trace(
            go.Scatter3d(
                x=traj[:, 0], y=traj[:, 1], z=traj[:, 2],
                mode='lines',
                name=body.name,
                line=dict(
                    color=color,
                    width=4 if body.name in ["Sun", "Rogue Star"] else 2
                ),
                opacity=0.8
            ),
            row=1, col=1
        )
        
        # Final position marker
        fig.add_trace(
            go.Scatter3d(
                x=[traj[-1, 0]], y=[traj[-1, 1]], z=[traj[-1, 2]],
                mode='markers',
                marker=dict(
                    size=PLANET_SIZES.get(body.name, 8),
                    color=color
                ),
                showlegend=False
            ),
            row=1, col=1
        )
    
    # Add debris to 3D plot
    if show_debris and debris_bodies:
        debris_final_x = [b.position[0] for b in debris_bodies]
        debris_final_y = [b.position[1] for b in debris_bodies]
        debris_final_z = [b.position[2] for b in debris_bodies]
        debris_colors, _ = classify_debris_particles(debris_bodies)
        
        fig.add_trace(
            go.Scatter3d(
                x=debris_final_x, y=debris_final_y, z=debris_final_z,
                mode='markers',
                name='Debris',
                marker=dict(size=2, color=debris_colors, opacity=0.5)
            ),
            row=1, col=1
        )
    
    # Panel 2: Final eccentricity bar chart
    planet_names = list(final_elements.keys())
    eccentricities = [final_elements[p]['eccentricity'] for p in planet_names]
    bound_status = ['EJECTED' if not final_elements[p]['bound'] else '' for p in planet_names]
    bar_colors = [PLANET_COLORS.get(p, 'white') for p in planet_names]
    
    fig.add_trace(
        go.Bar(
            x=planet_names,
            y=eccentricities,
            marker_color=bar_colors,
            text=bound_status,
            textposition='outside',
            name='Eccentricity',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Add reference line at e=1.0 (escape threshold) using a shape
    # Note: add_hline doesn't work well with 3D subplot layouts
    fig.add_shape(
        type="line",
        x0=-0.5, x1=len(planet_names)-0.5,
        y0=1.0, y1=1.0,
        line=dict(color="red", width=2, dash="dash"),
        xref="x2", yref="y2"
    )
    
    # Add annotation for the line
    fig.add_annotation(
        x=len(planet_names)-1, y=1.1,
        text="Escape (e=1)",
        showarrow=False,
        font=dict(color="red", size=10),
        xref="x2", yref="y2"
    )
    
    # Panel 3: Timestep evolution (if adaptive)
    if show_timestep and len(sim.timesteps) > 1:
        timestep_times = times[:len(sim.timesteps)]
        
        fig.add_trace(
            go.Scatter(
                x=timestep_times,
                y=sim.timesteps,
                mode='lines',
                name='Timestep',
                line=dict(color='cyan', width=2),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # Calculate axis ranges
    all_positions = np.concatenate(
        [traj for name, traj in trajectories.items() 
         if not name.startswith('particle_')]
    )
    max_xy = max(
        abs(all_positions[:, 0].min()), abs(all_positions[:, 0].max()),
        abs(all_positions[:, 1].min()), abs(all_positions[:, 1].max())
    ) + 20
    max_z = max(abs(all_positions[:, 2].min()), abs(all_positions[:, 2].max())) + 10
    
    # Layout
    title = (
        f"Stellar Flyby Dashboard: {params['rogue_mass']:.2f} M☉ at "
        f"{params['impact_parameter']:.0f} AU, v={params['velocity_kms']:.0f} km/s"
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
    fig.update_scenes(
        xaxis=dict(
            title='X (AU)', backgroundcolor='#1a1a2e', gridcolor='gray',
            range=[-max_xy, max_xy]
        ),
        yaxis=dict(
            title='Y (AU)', backgroundcolor='#1a1a2e', gridcolor='gray',
            range=[-max_xy, max_xy]
        ),
        zaxis=dict(
            title='Z (AU)', backgroundcolor='#1a1a2e', gridcolor='gray',
            range=[-max_z, max_z]
        ),
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


# Main Visualization Function

def visualize_simulation_results(
    sim: InteractiveSimulation,
    params: dict,
    show_debris: bool = True,
    save_html: bool = True,
    output_prefix: str = "simulation"
) -> Dict[str, go.Figure]:
    """
    Below is all visualizations for a completed simulation
    This is the main function to call after running your simulation
    
    Args:
        sim: Completed InteractiveSimulation object
        params: Dictionary of simulation parameters from get_user_parameters()
        show_debris: Whether to show debris disk (toggle on/off)
        save_html: Whether to save HTML files
        output_prefix: Prefix for output file names
        
    Returns:
        Dictionary of all figure objects
    """
    print("\n" + "=" * 60)
    print("Creating Visualizations")
    print("=" * 60)
    print(f"Debris disk visualization: {'ON' if show_debris else 'OFF'}")
    
    figures = {}
    
    # Static 3D plot
    print("\n1. Creating interactive 3D plot...")
    figures['static_3d'] = create_interactive_3d_plot(sim, params, show_debris)
    
    # Animated 3D plot
    print("2. Creating animated 3D plot with time slider...")
    figures['animated_3d'] = create_animated_3d_plot(sim, params, show_debris)
    
    # Debris comparison (if debris enabled)
    debris_bodies = [b for b in sim.bodies if b.is_test_particle]
    if show_debris and debris_bodies:
        print("3. Creating debris before/after comparison...")
        figures['debris_comparison'] = create_debris_comparison_plot(sim, params)
    
    # Comprehensive dashboard
    print("4. Creating comprehensive dashboard...")
    figures['dashboard'] = create_comprehensive_dashboard(
        sim, params, 
        show_debris=show_debris,
        show_timestep=params.get('adaptive', True)
    )
    
    # Save HTML files
    if save_html:
        print("\nSaving HTML files...")
        
        figures['static_3d'].write_html(f"{output_prefix}_3d.html")
        print(f"  ✓ {output_prefix}_3d.html")
        
        figures['animated_3d'].write_html(f"{output_prefix}_animated.html")
        print(f"  ✓ {output_prefix}_animated.html")
        
        if 'debris_comparison' in figures:
            figures['debris_comparison'].write_html(f"{output_prefix}_debris.html")
            print(f"  ✓ {output_prefix}_debris.html")
        
        figures['dashboard'].write_html(f"{output_prefix}_dashboard.html")
        print(f"  ✓ {output_prefix}_dashboard.html")
    
    print("\n✓ Visualization complete!")
    print("  Open the HTML files in your browser to interact with the plots.")
    
    return figures


# Example Usage / Main

if __name__ == "__main__":
    """
    Example: Run a simulation and visualize the results
    
    This demonstrates how to use this module with interactive_simulator.py
    """
    print("=" * 60)
    print("Enhanced Visualization Demo")
    print("=" * 60)
    
    # Default parameters (normally from get_user_parameters())
    params = {
        'rogue_mass': 0.7,
        'impact_parameter': 35.0,
        'velocity': 40.0 * KM_S_TO_AU_YR,
        'velocity_kms': 40.0,
        'inclination': np.radians(5.0),
        'include_inner': True,
        'include_outer': True,
        'planet_masses': {},
        'add_debris': True,
        'n_debris': 300,
        'duration': 100.0,
        'integrator': 'leapfrog',
        'adaptive': True,
        'base_dt': 0.1
    }
    
    # Create solar system with all planets
    print("\nSetting up simulation...")
    solar_system = create_solar_system(
        include_inner=params['include_inner'],
        include_outer=params['include_outer'],
        planet_mass_multipliers=params['planet_masses']
    )
    
    # Create rogue star
    rogue_star = create_rogue_star(
        mass=params['rogue_mass'],
        impact_parameter=params['impact_parameter'],
        velocity_infinity=params['velocity'],
        inclination=params['inclination']
    )
    
    all_bodies = solar_system + [rogue_star]
    
    # Add debris disk
    if params['add_debris']:
        debris = create_debris_disk(n_particles=params['n_debris'])
        all_bodies.extend(debris)
        print(f"Added {len(debris)} debris particles")
    
    # Run simulation
    print(f"\nRunning simulation for {params['duration']} years...")
    sim = InteractiveSimulation(
        bodies=all_bodies,
        base_dt=params['base_dt'],
        integrator=params['integrator'],
        adaptive=params['adaptive']
    )
    
    sim.run(duration=params['duration'], save_interval=0.2, verbose=True)
    
    # Print orbital outcomes
    print("\n" + "=" * 60)
    print("FINAL ORBITAL STATE")
    print("=" * 60)
    
    final_elements = sim.get_final_orbital_elements()
    for planet, elem in final_elements.items():
        status = "BOUND" if elem['bound'] else "EJECTED!"
        print(f"  {planet:10s}: a={elem['semi_major_axis']:8.2f} AU, "
              f"e={elem['eccentricity']:.3f} [{status}]")
    
    # Create visualizations with debris ON
    figures = visualize_simulation_results(
        sim, params,
        show_debris=True,   # Toggle this to turn debris on/off
        save_html=True,
        output_prefix="flyby_demo"
    )
    
    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - flyby_demo_3d.html         : Interactive 3D view")
    print("  - flyby_demo_animated.html   : Animation with time slider")
    print("  - flyby_demo_debris.html     : Before/after debris comparison")
    print("  - flyby_demo_dashboard.html  : Comprehensive dashboard")
