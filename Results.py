# Author: Tessa van Duin
# Based on code by: Rik Broekhoven
#
# Date: 26/03/25

from ClassManager import EnergySpectrumManager
from Back_end_ClassManager import (
    simple_square_lattice, 
    simple_hexagonal_lattice,
    manual_lattice
)
import numpy as np

# distances 
d = 5/2

# structure
structure = simple_square_lattice(2, 2 , d)
N = np.shape(structure)[0]
S = [1/2]*N  # Spin quantum numbers

# constants
h = 6.62607015e-34  # Planck constant [Js]
hbar = h / (2 * np.pi)  # Reduced Planck constant [Js/rad]
muB = 9.27400915e-24  # Bohr magnetorn [J/T]
g = np.array([1.74]*N)   #wat???
gyro = (g * muB) / hbar  # Example gyromagnetic ratios for 4 spins
Btip = np.array([0,0,0])  # Example tip magnetic field (T)
B_hat = np.array([0, 0, 1])
B = 0.00009 * B_hat



# Initialize the manager
manager = EnergySpectrumManager(structure=structure, gyro=gyro, N=N, S=S, B=B, Btip=Btip)

# Plot eigenvalues
manager.run_and_plot(B=B, tolerance=1e-12, d=d)
manager.plot_eigenstate_complex_contributions(d, B)

# Plot exchange and dipolar coupling decay
distances = np.linspace(1,4,7)
manager.plot_interaction_decay(distances, shape='Hex', tolerance=1e-18)


# Define values for structure decay due to Kondo Rates
T = 0.3
t_values = np.linspace(0, 200, 1000)
manager.plot_structure_decay(T=T, B=B, dist=d, flip_spin=False, coherence=False, times=t_values, assymmetric=True, reset=True)