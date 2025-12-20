# Fall 2025 Computational Astrophysics (A student's repo breakdown)

A collection of coursework for my Computational Methods in Astrophysics class, including homework assignments and a final project on stellar flyby simulations.

## Repository Structure

### Homework Assignments

| File | Topic |
|------|-------|
| `compastro.py` | Homework 1 |
| `hw2.py` | Homework 2 |
| `hw3.py` | Homework 3 |
| `hw4.py` | Homework 4 |
| `hw5.py` | Homework 5 |
| `hw6.py` | Homework 6 |

**Data files:**
- `tic0011480287.fits` - Hw 4 TESS data file

**Supporting personal class exercises:**
- `exercise_4.2.py` - Exercise from textbook/course materials
- `difference.py` - Numerical differentiation methods
- `precision.py` - Numerical precision analysis

---

### Final Project: Stellar Flyby Simulation

Simulating the gravitational effects of a stellar flyby on a planetary system using N-body dynamics.

**Project folder:**
- `Final Project/` - Additional final project materials
**Main simulation code:**
- `stellar_flyby_simulation.py` - Core N-body simulation of stellar flyby event (Simple Version)
- `interactive_simulator.py` - Expanded Final Project (See top comments for running instructions)
- `enhanced_interactive_visualization.py` - Expanded options for visualizating user parameter simulated senario
- `debris_disk_simulation.py` - Simulating debris disk perturbations during flyby (Supplemental)

**Visualizations Outputs:**

| File | Description |
|------|-------------|
| `simulation_results.html` | Summary of simulation results |
| `animated_3d.html` | 3D animated visualization of simulation |
| `interactive_3d.html` | Interactive 3D viewer |
| `flyby_demo_3d.html` | 3D flyby demonstration |
| `flyby_demo_animated.html` | Animated flyby visualization |
| `flyby_demo_dashboard.html` | Dashboard with simulation metrics |
| `flyby_demo_debris.html` | Debris disk visualization during flyby |
| `debris_disk_3d.html` | 3D debris disk viewer |
| `debris_before_after.html` | Comparison of system before/after flyby |
| `dashboard.html` | Main simulation dashboard |


**Output figures:**
- `debris_comparison.png` - Before/after comparison of debris disk
- `orbital_evolution.png` - Evolution of orbital parameters over time
- `trajectories_3d.png` - 3D trajectory plot of system bodies

---
## Requirements

```
numpy
matplotlib
scipy
plotly  # for interactive HTML visualizations
astropy # for FITS file handling
```

## Reference
- https://compmeth.commons.gc.cuny.edu/syllabus/ - Class Website
- https://github.com/ari-maller/comp_meth_notebooks/tree/main - Reference Examples

## Author
Lisha Ramon

## Acknowledgments
Special thanks to Professor Ari Maller for their guidance and instruction throughout this course.
