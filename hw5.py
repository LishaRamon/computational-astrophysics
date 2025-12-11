# HW 5 - Monte Carlo Radiative Transfer
"""
One of the most common uses of Monte Carlo calculations in astrophysics is for radiative transfer. 
The simplest example of this is the motion of radiation in a star. 
The cores of stars are hot and dense, 
such that all electrons are ionized and the cross section for most photons to interact with an electron 
is just the frequency independent Thompson cross-section.

\sigma_T = \frac{8\pi}{3} \big(\frac{e^2}{4\pi \epsilon_0 m_e c^2}\big)^2  = 6.652 \times10^{-25} cm^2 

Another import quantity to consider in scattering theory is the mean free path of a particle.

l = 1/(n \times\sigma) 

If n_e = 10^{20} cm^{-3} the mean free path would be 150m. 
Start by just considering a slab of constant electron density with this value and a width of 1km. 
Perform a Monte Carlo simulation of a photon through this slab. 
Note that for low energy photons (E = hf << m_e c^2) the photon scattering is isotropic, equally likely in any direction. 
However, since you are only trying to follow the photons progress through the slab, you only need to consider the 
angle the scattered photon makes with the direction through the slab. 
Also note that your photon is almost as likely to be reflected back to the beginning as make it through 
the slab, so make sure you account for this possibility. 
Make an animation of your photons path. One example of how to make an animation can be found here.

Now to extend this calculation to the Sun we need to know the Sun’s electron density as a function of radius. 
A reasonably accurate fit is given as equation 4.2 in Neutrino Astrophysics by John Bachall. 
The formula is

n_e(r) = 2.5 \times 10^{26} exp^{-\frac{r}{0.096 R_{sun}}}  cm^{-3}

where R_{sun} is the solar radius, 696,340km. 
The formula is only good to 0.9R_{sun}, after which density drops quickly and one can assume there are 
no more scatterings for photons. We can now calculate the time it takes for a photon to escape the Sun. 
Note that the mean free path changes by many orders of magnitude as the radius changes. 
So you can’t simply ask what is the probability of a photon scattering for some fixed distance, like 1cm. 
Instead, just like for radioactive decay, we can draw how far the photon traveled before a scattering from 
the exponential distribution.

P(x) = \frac{1}{l} e^{-x/l} 

So to simulate the photons path, start at r=0, calculate the electron density and from that the mean free path.
Draw a random number from the exponential distribution with a scale of the mean free path. And update r. 
Calculate the new electron density, the new mean free path and the angle the photon was scatter by 
(just 0 to 180 because we only care about relative to r) and then draw a new random number from the 
exponential distribution with the new mean free path. 
Note you must keep track of both r, the distance from the Sun’s center and s, the total distance traveled 
by the photon, as the total distance is the time, t = cs. """

"""
Part 1: Photon Transport Simulation

Simulate photon transport through a slab with constant electron density
   - n_e = 10^20 cm^-3, mean free path = 150m, slab width = 1km
   - Track reflection, transmission, absorption
   - Make an animation of photon paths


The simulation tracks photon paths, scattering, absorption, reflection, and transmission through different media using Monte Carlo methods.
Includes slab geometry and stellar (Sun) transport models.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import random 
import matplotlib.animation as animation

#Part 1: Photon through a slab of constant electron density

def simulate_photon(mean_free_path=150, slab_width=1000, albedo=1.0):
    """
    Simulate a single photon through a slab.
    
    Parameters:
    - mean_free_path: Average distance between scatters (meters)
    - slab_width: Thickness of slab (meters)
    - albedo: Probability of scattering vs absorption (1.0 = no absorption)
    
    Returns: path_x, path_y, fate ('transmitted', 'reflected', 'absorbed')
    """
    # begin at slab enterance
    x, y = 0.0, 0.0 #initial position in an x,y plane
    path_x = [x] #list to store x positions
    path_y = [y] #list to store y positions
    
    # first photon moves forward into slab
    first_step = True
    max_steps = 10000
    
    for step in range(max_steps):
        # show random distance from exponential distribution
        # P(r) = (1/λ) * exp(-r/λ) -> sample using inverse transform
        r = -mean_free_path * np.log(1 - random.random())
        
        # Draw random direction
        if first_step:
            # First step: forward direction only (-90° to +90°)
            theta = random.uniform(-np.pi/2, np.pi/2)
            first_step = False
        else:
            # after scatter: isotropic (any direction)
            theta = 2 * np.pi * random.random()
        
        # find new position
        x_new = x + r * np.cos(theta)
        y_new = y + r * np.sin(theta)
        
        # check boundaries
        if x_new > slab_width:
            # if photon transmitted through slab
            path_x.append(x_new)
            path_y.append(y_new)
            return path_x, path_y, 'transmitted'
        
        if x_new < 0:
            # if photon reflected back
            path_x.append(x_new)
            path_y.append(y_new)
            return path_x, path_y, 'reflected'
        
        # update position - photon still in slab
        x, y = x_new, y_new
        path_x.append(x)
        path_y.append(y)
        
        # check for absorption (if albedo < 1)
        if random.random() > albedo:
            return path_x, path_y, 'absorbed'
    
    # max steps reached
    return path_x, path_y, 'absorbed'


"""
    Run full slab simulation with multiple photons
    
    n_e = 10^20 cm^-3 gives mean free path = 150m, slab width = 1km

    Parameters:
    - n_photons: Number of photons to simulate
    - mean_free_path: Mean free path in meters
    - slab_width: Slab thickness in meters  
    - albedo: Scattering probability (0.9 = 90% scatter, 10% absorb)
    """
def slab_simulation(
        
    n_photons=1000, 
    mean_free_path=150, 
    slab_width=1000, 
    albedo=0.9):

    transmitted = 0
    reflected = 0
    absorbed = 0
    all_paths = []
    
    print(f"Simulating {n_photons} photons through slab...")
    print(f"Slab width: {slab_width}m, Mean free path: {mean_free_path}m")
    
    for i in range(n_photons):
        # check progress
        if (i + 1) % (n_photons // 10) == 0:
            print(f"Progress: {100*(i+1)/n_photons:.0f}%")
        
        # simulate single photon
        path_x, path_y, fate = simulate_photon(mean_free_path, slab_width, albedo)
        
        # Store first 50 paths for animation
        if len(all_paths) < 50:
            all_paths.append((path_x, path_y, fate))
        
        # Keep track of outcomes
        if fate == 'transmitted':
            transmitted += 1
        elif fate == 'reflected':
            reflected += 1
        else:
            absorbed += 1
    
    # Print results
    print("\n" + "="*50)
    print("Slab Simulation Results")
    print("="*50)
    print(f"Total photons: {n_photons}")
    print(f"Transmitted: {transmitted} ({100*transmitted/n_photons:.1f}%)")
    print(f"Reflected: {reflected} ({100*reflected/n_photons:.1f}%)")
    print(f"Absorbed: {absorbed} ({100*absorbed/n_photons:.1f}%)")
    print("="*50)
    
    return all_paths

# Animation function photon paths through slab
"""
Parameters:
    - photon_paths: List of (path_x, path_y, fate) tuples
    - mean_free_path: For display purposes
    - slab_width: Slab width in meters
"""
def animate_photons(photon_paths, mean_free_path=150, slab_width=1000):
    print("\nProducing animation...")
    
    # picking first 20 paths for animation
    paths_to_animate = photon_paths[:20]
    
    # set plot limits
    all_x = []
    all_y = []
    for path_x, path_y, _ in paths_to_animate:
        all_x.extend(path_x)
        all_y.extend(path_y)
    
    # initiaze figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # set limits
    ax.set_xlim(min(all_x) - 200, max(all_x) + 200)
    ax.set_ylim(min(all_y) - 100, max(all_y) + 100)
    
    # show slab boundaries
    ax.axvline(x=0, color='black', linewidth=2, label='Slab Entrance')
    ax.axvline(x=slab_width, color='black', linewidth=2, label='Slab Exit')
    
    # add shaded region for slab distinction
    ax.fill_between([0, slab_width], ax.get_ylim()[0], ax.get_ylim()[1], 
                    alpha=0.1, color='black')
    
    # labels
    ax.set_xlabel('Position (m)', fontsize=12)
    ax.set_ylabel('Lateral Position (m)', fontsize=12)
    ax.set_title(f'Photon Random Walk Animation\nSlab Width: {slab_width}m, Mean Free Path: {mean_free_path}m', 
                fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # set colors for each photon
    colors = plt.cm.rainbow(np.linspace(0, 1, len(paths_to_animate)))
    
    # set photon markers and trails
    photon_circles = []
    trail_lines = []
    
    for i, color in enumerate(colors):
        # circle for photon position
        circle = plt.Circle((0, 0), 20, fc=color, alpha=0.8)
        photon_circles.append(circle)
        
        # line for trail
        line, = ax.plot([], [], color=color, linewidth=1, alpha=0.5)
        trail_lines.append(line)
    
    # find maximum path length
    max_steps = max(len(path_x) for path_x, _, _ in paths_to_animate)
    

    # Start animation state
    def init():

        for circle in photon_circles:
            ax.add_patch(circle)
        for line in trail_lines:
            line.set_data([], [])
        return photon_circles + trail_lines
    
    def animate(frame):
        # animation function for each frame
        for i, (circle, line, (path_x, path_y, fate)) in enumerate(
                zip(photon_circles, trail_lines, paths_to_animate)):
            
            if frame < len(path_x):
                # update photon position
                circle.center = (path_x[frame], path_y[frame])
                # update trail
                line.set_data(path_x[:frame+1], path_y[:frame+1])
            else:
                # photon finished - show final position
                circle.center = (path_x[-1], path_y[-1])
                line.set_data(path_x, path_y)
                
                # fade absorbed photons
                if fate == 'absorbed':
                    circle.set_alpha(0.3)
        
        return photon_circles + trail_lines
    
    # create animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                  frames=max_steps, interval=50,
                                  blit=True, repeat=True)
    
    # Save as gif
    print("Saving animation as 'photon_paths.gif'...")
    writer = animation.PillowWriter(fps=20)
    anim.save('photon_paths.gif', writer=writer)
    print("Animation saved!")
    
    plt.show()
    return anim


""" 
Part 2: Simulate Photon's Escaping from Sun's Core
   - Use density profile: n_e(r) = 2.5e26 * exp(-r/(0.096*R_sun)) cm^-3
   - Calculate time for photon to escape from center to surface
"""

def simulate_solar_photon(max_steps=1000000):
  
    # constants
    R_sun = 6.96e10  # Solar radius in cm
    c = 3e10  # Speed of light in cm/s
    sigma_T = 6.652e-25  # Thomson cross-section in cm^2
    
    # start at Sun's center
    r = 0.0  # Current radius
    s = 0.0  # Total distance traveled
    n_scatters = 0
    
    print("Starting photon at Sun's core...")
    
    for step in range(max_steps):
        # find electron density at current radius
        # n_e(r) = 2.5e26 * exp(-r/(0.096*R_sun)) cm^-3
        n_e = 2.5e26 * np.exp(-r / (0.096 * R_sun))
        
        # find mean free path
        # l = 1/(n_e * sigma_T)
        l = 1.0 / (n_e * sigma_T)
        
        # show random distance from exponential distribution
        # P(x) = (1/l) * exp(-x/l)
        distance = -l * np.log(1 - random.random())
        
        # update total distance traveled
        s += distance
        
        # draw random scattering angle (0 to 180 degrees)
        # For isotropic scattering, cos(theta) is uniform in [-1, 1]
        cos_theta = 2 * random.random() - 1
        
        # update radius using law of cosines
        # r_new^2 = r^2 + d^2 + 2*r*d*cos(theta)
        r_new = np.sqrt(r**2 + distance**2 + 2*r*distance*cos_theta)
        r = r_new
        
        n_scatters += 1
        
        # print progress occasionally
        if step % 100000 == 0 and step > 0:
            print(f"  Step {step}: r/R_sun = {r/R_sun:.3f}")
        
        # check if photon escaped (reached surface)
        if r >= R_sun:
            # calculate escape time
            time_seconds = s / c
            time_years = time_seconds / (365.25 * 24 * 3600)
            
            print("\n" + "="*50)
            print("photon has found it's way out!")
            print("="*50)
            print(f"Number of scatters: {n_scatters:,}")
            print(f"Total distance traveled: {s:.2e} cm")
            print(f"  = {s/R_sun:.1f} solar radii")
            print(f"Time to escape: {time_years:,.0f} years")
            print("="*50)
            
            return True, n_scatters, s, time_years
    
    print(f"Photon still trapped after {max_steps:,} steps")
    print(f"Current position: r/R_sun = {r/R_sun:.3f}")
    return False, n_scatters, s, None



# main Function

def main():
    
    # Part 1: Slab simulation with animation
    print("\nPart 1: Photon transport through slab")
    print("-" * 40)
    
    # Use parameters: n_e = 10^20 cm^-3, mean free path = 150m, width = 1km
    photon_paths = slab_simulation(
        n_photons=10000,
        mean_free_path=150,  # 150 meters
        slab_width=1000,     # 1 kilometer
        albedo=0.9          # probability that photon scatters at interaction, leaves opportunity for absorbtion
                            # 90% scatter, 10% absorb
    )
    
    # Create required animation
    animate_photons(photon_paths, mean_free_path=150, slab_width=1000)
    
    # Part 2: Solar photon escape
    print("\nPart 2: Photon escapes from sun core")
    print("-" * 40)
    
    escaped, n_scatters, distance, time_years = simulate_solar_photon()
    if not escaped:
        print("Photon did not escape the Sun within the maximum steps.")

if __name__ == "__main__":
    main()