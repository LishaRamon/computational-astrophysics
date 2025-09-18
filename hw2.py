# Assignment 2
# HW2 – Exercise 5.3: Integration
# Consider the integral
# E(x) = \int^x_0 e^{-t^2} dt  
# a) Write a program to calculate E(x) for values of x from 0 to 3 in steps of 0.1. Choose for yourself what method you will use for performing the integral and a suitable number of slices.
# b) When you are convinced your program is working, extend it further to make a graph of E(x) as a function of x.
# Note that there is no known way to perform this particular integral analytically, so numerical approaches are the only way forward.


import numpy as np
import matplotlib.pyplot as plt


# Function to use trapezoidal rule to calculate E(x) = integral from 0 to x of e^(-t^2) dt

def E(x, num_slices=10000):
    """
    This function calculated the E(x) using the trapezoidal rule
    
    x: shows the upper limit of integral
    num_slices: shows how many 'slices' we cut the integral into (more = more accurate)
    """
    
    # when x is 0, integral is 0
    if x == 0:
        return 0
    
    # first: define the t values from 0 to x
    slice_size = x / num_slices
    t_values = []
    for i in range(num_slices + 1):
        t = i * slice_size
        t_values.append(t)
    
    # second: calculate e^(-t^2) for each t value
    int_values = []
    for t in t_values:
        y = np.exp(-t**2)  # translates to e^(-t^2)
        int_values.append(y)
    
    # three: use trapezoidal rule
    # formula: (slice_size/2) * [f(0) + 2*f(1) + 2*f(2) + ... + 2*f(n-1) + f(n)]
    
    integral = 0
    
    # add first point (counting it only one time)
    integral += int_values[0]
    
    # add middle points (counting them twice)
    for i in range(1, num_slices):
        integral += 2 * int_values[i]
    
    # add last point (same as the first point, count it only one time)
    integral += int_values[num_slices]
    
    # then multiply by slice_size/2
    integral = integral * slice_size / 2
    
    return integral

# part A: calculate E(x) for x from 0 to 3 in steps of 0.1
print("Calculating E(x) = integral of e^(-t^2) from 0 to x")
print("Using 10000 slices for accuracy")

# creating the empty lists for our x values
x_list = []
e_list = []

x = 0.0
while x <= 3.0:
    x_list.append(x)
    e_value = E(x, 10000)
    e_list.append(e_value)
    
    print(f"x = {x:.1f}, E(x) = {e_value:.6f}")
    
    x += 0.1  # next x value

# part B: graph it
plt.figure(figsize=(15, 6))
plt.plot(x_list, e_list, linewidth=2, marker='*', markersize=6, color='purple')
plt.xlabel('x-value')
plt.ylabel('E(x)')
plt.title('Graph for E(x) = ∫₀ˣ e^(-t²) dt')
plt.grid(True)

# extra text for graph visuals
plt.text(2, 0.3, f'At x=3: E(3) = {e_list[-1]:.4f}', 
         bbox=dict(boxstyle="round", facecolor='yellow'))

plt.show()

print("Finishing notes! Presented graph shows how E(x) grows as x increases")
print("See how growth slows down, due to e^(-t^2) becoming smaller for larger t")

#Claude used for clarification