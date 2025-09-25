# Assignment 3
# HW3 – Exercise 6.16: The Lagrange Point
# There is a magical point between the Earth and the Moon, called the L1 Lagrange point, 
# at which a satellite will orbit the Earth in perfect synchrony with the Moon, staying always in between the two. 
# This works because the inward pull of the Earth and the outward pull of the Moon combine to create exactly the needed centripetal force that keeps the satellite in its orbit. 
# Here’s the setup:

#Setup of L1 Lagrange point
# Assuming circular orbits, and assuming that the Earth is much more massive than either the Moon or the satellite, show that the distance r from the center of the Earth to the L1 point satisfies

# GM/r² - Gm/(R-r)² = ω²r,  

# where M and m are the Earth and Moon masses, G is Newton’s gravitational constant, and ω is the angular velocity of both the Moon and the satellite.
#The equation above is a fifth-order polynomial equation in r (also called a quintic equation). 
# Such equations cannot be solved exactly in closed form, but it’s straightforward to solve them numerically. 
# Write a program that uses either Newton’s method or the secant method to solve for the distance r from the Earth to the L1 point. 
# Compute a solution accurate to at least four significant figures.
#The values G and the Earth’s mass can be found in the astropy.constants or scipy.constants sub-packages. 
# The Moon’s mass is 7.348e22 kg and the Earth-Moon distance is 3.844e8 m. 
# The value of ω is 2.662e-6 s-1. You will also need to choose a suitable starting value for r, or two starting values if you use the secant method.
# used Claude  for clarification on the physics and equation setup
"""
    Using newton's method to find L1 Lagrange point    
    The equation to solve is: GM/r² (Earth's pull) - Gm/(R-r)² (Moon's pull) = ω²r (centripetal force for Moon-speed orbit)
    Rearranged as: M/r² - m/(R-r)² = ω²r/G = 0 for Newton's method
    """
"""
r = distance from Earth to our satellite (the L1 point)
"""

def lagrange_l1():
    
    G = 6.674e-11  # Gravitational constant (m³/kg⋅s²)
    earth_mass = 5.972e24 # Earth mass (kg)
    moon_mass = 7.348e22  # Moon mass (kg)
    R = 3.8446e8 # Earth to Moon distance (m) or (384,460 km)
    ω = 2.662e-6 # Angular velocity/speed (rad/s | how fast the Moon orbits)
    # state f(r) want to find (distance from Earth to L1 point)
    
    # f(r) = M/r² - m/(R-r)² - ω²r/G = 0
    def f(r):
        earth_pull = earth_mass / (r**2)
        moon_pull = moon_mass / ((R - r)**2)
        centriforce = (ω**2 * r) / G
        return earth_pull - moon_pull - centriforce
    
    # f'(r) (derivative) for Newton's method
    def deriative(r):
        earth_pull = -2 * earth_mass / (r**3)
        moon_pull = -2 * moon_mass / ((R - r)**3)
        centriforce = (ω**2) / G
        return earth_pull + moon_pull - centriforce
    
    # newton method function
    def newton_method(first_guess, tolerance=1e-6, max_iterations=100):
        r = first_guess
        
        for i in range(max_iterations):
            dist_from_zero = f(r)
            change_rate = deriative(r)
            
            if abs(dist_from_zero) < tolerance:
                print(f"Converged (solution found) after {i+1} cycles")
                return r
            
            if abs(change_rate) < 1e-15:  # avoiding % by 0
                print("Derivative too small, stopping")
                break
                
            r_new = r - dist_from_zero / change_rate # r_new is next guess where L1 point is
            
            print(f"Cycle {i+1}: r = {r:.6e} m, f(r) = {dist_from_zero:.6e}")
            
            r = r_new 
        
        print(f"Did not converge after {max_iterations} cycles")
        return r
    
    # pick a first guess - L1 point is between Earth and Moon, closer to Earth
    # good starting point is about 85% of the way from Earth to Moon (based on research)
    initial_r = 0.85 * R
    
    print("Solving for L1 Lagrange point")
    print(f"Earth mass: {earth_mass:.3e} kg")
    print(f"Moon mass: {moon_mass:.3e} kg") 
    print(f"Earth-Moon distance: {R:.3e} m")
    print(f"Angular velocity: {ω:.3e} rad/s")
    print(f"First guess: {initial_r:.3e} m")
    print()
    
    # use newtons method here | r_l1 is Lagrange point distance from Earth center
    r_l1 = newton_method(initial_r)
    
    print()
    print("Final outputs:")
    print(f"L1 point distance from Earth center: {r_l1:.3e} m")
    print(f"L1 point distance from Earth center: {r_l1/1000:.1f} km")
    print(f"Distance from Earth surface: {(r_l1 - 6.371e6)/1000:.1f} km")
    print(f"Fraction of Earth-Moon distance: {r_l1/R:.4f}")
    
    # check solution
    check = f(r_l1)
    print(f"Check f(r_L1) = {check:.6e} (should be ~0)")
    
    return r_l1

# Run the calculation
r_l1 = lagrange_l1()