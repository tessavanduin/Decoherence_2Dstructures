# Author: Tessa van Duin
# Based on code by: Rik Broekhoven
#
# Date: 26/03/25


import itertools
import numpy as np
from qutip import  tensor, qeye, spin_Jx, spin_Jy, spin_Jz, basis, sigmax, sigmay, sigmaz
from itertools import combinations
import math
from collections import defaultdict
from scipy import linalg

sigmas = [sigmax(), sigmay(), sigmaz()]
k_B = 8.617E-5 * 2.418E+5 # GHz/K
e = 2.418E+5 # GHz/eV 

def simple_square_lattice(rows, cols, dist):
    x = np.linspace(0, (cols - 1) * dist, cols)
    y = np.linspace(0, (rows - 1) * dist, rows)
    xx, yy = np.meshgrid(x, y)
    
    chain = np.zeros((rows * cols, 3))
    chain[:, 0] = xx.flatten()  # x-coordinates
    chain[:, 1] = yy.flatten()  # y-coordinates
    
    print(chain)
    return chain

def placement_sites(latticepoints):
    
    def evenodd(num):
        if num % 2 == 0:
            return 'even' 
        else:
            return 'odd'
        
    placement_sites = []
    for (x, y) in latticepoints:
        site = evenodd(x+y)
        placement_sites.append(site)
    return placement_sites

def simple_hexagonal_lattice(layers, dist, middle=True):
    """    
    Parameters:
        layers (int): Number of hexagonal layers around the center.
        dist (float): Distance between adjacent atoms.
        middle (bool): Whether to include the central atom.
    Returns:
        np.ndarray: Coordinates of the atoms in the hexagonal lattice.
    """
    lattice_points = []

    # Define directions for hexagonal structure
    directions = [
        (1, np.sqrt(3)),    # Right-Up
        (-1, np.sqrt(3)),   # Left-Up
        (-2, 0),            # Left
        (-1, -np.sqrt(3)),  # Left-Down
        (1, -np.sqrt(3)),   # Right-Down
        (2, 0)              # Right
    ]
    
    # Generate points for each layer
    for layer in range(1, layers + 1):
        # Start at the starting position of the layer (right-down, clock-wise)
        x, y = layer * directions[4][0] * dist, layer * directions[4][1] * dist
        
        # Add points in each direction
        for direction in directions:
            for _ in range(layer):
                lattice_points.append((x, y))
                # Move to the next position
                x += direction[0] * dist
                y += direction[1] * dist

    # Add the central point if middle is True
    if middle:
        lattice_points.append((0, 0))

    # Convert to numpy array
    lattice_points = np.array(lattice_points)
    
    # Center the lattice by subtracting the mean of all points
    center = np.mean(lattice_points, axis=0)
    lattice_points -= center

    # Round for numerical stability
    lattice_points = np.round(lattice_points, decimals=0)
    print(f'Hexagon configuration for distance {dist*2} is: {placement_sites(lattice_points)}')
    
    # Gather the different angles between the atoms in the hexagon
    angles = []  
    for (x,y) in lattice_points:
        if x != 0:
            angles.append((np.arctan(y/x)*180/math.pi))   #entries in the array with the different angles

    print(f"Hexagon approximate angle for distance {dist*2} is: {angles}")
    print(f"Latticepoints are {lattice_points}")
    return lattice_points

def manual_lattice():
    """
    Allows the user to manually input spin coordinates (x, y, optional z).
    Stops when the user types "done".
    
    Returns:
        np.ndarray: Array of manually entered coordinates.
    """
    lattice_points = []
    
    print("Enter spin coordinates one by one (format: x y [z]) and click enter after each coordinate. Type 'done' when finished.")

    while True:
        user_input = input("Enter coordinate: ").strip()

        if user_input.lower() == "done":
            break  # Exit input loop

        try:
            coords = list(map(float, user_input.split()))  # Convert input to floats
            if len(coords) not in [2, 3]:
                print("Error: Enter 2D (x y) or 3D (x y z) coordinates.")
                continue

            lattice_points.append(coords)

        except ValueError:
            print("Invalid input. Please enter numbers separated by spaces.")

    # Convert to NumPy array
    lattice_array = np.array(lattice_points)
    
    print("Final spin structure:\n", lattice_array)
    return lattice_array



def h_ident(N=2, S=[1 / 2]):
    """ Help function
    Generates tensor product identity of dimension of all spins in the system
      
    Parameters:
    N: int
        amount of spins
    S: float
        spin quantum value
     
    """
    return [qeye(int(2 * S[n] + 1)) for n in range(N)]  # Ensure each element is a Qobj

def hamiltonian(lattice, larmor, J, D, N=2, S=[1/2]):
    """
    Build 2D spin Hamiltonian for a lattice.
    """
    if larmor.shape[0] != N or larmor.shape[1] != 3:
        raise ValueError("Larmor frequencies must be of shape (N, 3) where N is the number of spins.")

    H = 0

    # Generate all atom combinations
    atomcomb = np.array(list(combinations(range(N), 2)))

    # Zeeman Interaction
    for i in range(N):
        for d in range(3):  # x, y, z components
            H += -larmor[i, d] * 2 * np.pi * spin_operators(i, N, S)[d]

    for i, (atom1, atom2) in enumerate(atomcomb):
        H_heis = [qeye(int(2 * S[n] + 1)) for n in range(N)]
        H_heis[atom1] = spin_Jx(S[atom1])
        H_heis[atom2] = spin_Jx(S[atom2])
        interaction_term = tensor(*H_heis)
        H += (J[i] + D[i]) * interaction_term * 2 * np.pi

    for i, (atom1, atom2) in enumerate(atomcomb):
        H_heis = [qeye(int(2 * S[n] + 1)) for n in range(N)]
        H_heis[atom1] = spin_Jy(S[atom1])
        H_heis[atom2] = spin_Jy(S[atom2])
        interaction_term = tensor(*H_heis)
        H += (J[i] + D[i]) * interaction_term * 2 * np.pi
    
    for i, (atom1, atom2) in enumerate(atomcomb):
        H_heis = [qeye(int(2 * S[n] + 1)) for n in range(N)]
        H_heis[atom1] = spin_Jz(S[atom1])
        H_heis[atom2] = spin_Jz(S[atom2])
        interaction_term = tensor(*H_heis)
        H += (J[i] + D[i]) * interaction_term * 2 * np.pi

    # Compute unit vectors n_ab for anisotropic term
    dim = lattice.shape[1]  # Detect lattice dimensionality (2 for 2D, 3 for 3D)
    n_ab = np.zeros((len(atomcomb), 3))  # Preallocate n_ab with 3D compatibility
    for i, (atom1, atom2) in enumerate(atomcomb):
        diff = lattice[atom2] - lattice[atom1]
        norm = np.linalg.norm(diff)
        if norm == 0:
            raise ValueError(f"Zero-length distance between atoms {atom1} and {atom2}. Check lattice coordinates.")
        n_ab[i, :dim] = diff / norm  # Update only the relevant dimensions


    # Anisotropic Interaction Hamiltonian
    for i, (atom1, atom2) in enumerate(atomcomb):
        # Compute the unit vector n_ab for the current pair (atom1, atom2)
        n_ab_vector = n_ab[i]  # This is atom1 3D vector (n_x, n_y, n_z)

        # Initialize the Hamiltonian term for this interaction pair
        H_aniso = [qeye(int(2 * S[n] + 1)) for n in range(N)]  # Identity for all spins

        # Compute the dot product for atom1
        H_aniso[atom1] = (
            n_ab_vector[0] * spin_Jx(S[atom1]) +  # n_x * Sx^{(atom1)}
            n_ab_vector[1] * spin_Jy(S[atom1]) +  # n_y * Sy^{(atom1)}
            n_ab_vector[2] * spin_Jz(S[atom1])    # n_z * Sz^{(atom1)}
        )
        
        # Compute the dot product for atom2
        H_aniso[atom2] = (
            n_ab_vector[0] * spin_Jx(S[atom2]) +  # n_x * Sx^{(atom2)}
            n_ab_vector[1] * spin_Jy(S[atom2]) +  # n_y * Sy^{(atom2)}
            n_ab_vector[2] * spin_Jz(S[atom2])    # n_z * Sz^{(atom2)}
        )
        
        # Compute the tensor product for the full Hamiltonian
        interaction_term = tensor(*H_aniso)

        # Add the interaction term to the Hamiltonian
        H += -3 * D[i] * interaction_term * 2 * np.pi

    return H

def spin_operators(l, N, S):
    """ Complete spin basis for N atoms of spin S
    -----------
    Parameters
    l: int
       index of atom
    N: int
       amount of spins
    S: float
       spin quantum number
    lattice_shape : tuple of int, optional
        Shape of the 2D lattice (rows, columns).
        
    Returns:
    list with complete spin basis for N atoms of spin S
    """
    # Ensure lattice_shape is compatible with N
        
    spin = [0]*3   
    
    spin_x = h_ident(N, S)
    spin_x[l] = spin_Jx(S[l]) 
    spin[0] = spin[0]+ tensor(spin_x)

    spin_y = h_ident(N, S)
    spin_y[l] = spin_Jy(S[l]) 
    spin[1] = spin[1]+ tensor(spin_y)

    spin_z = h_ident(N, S)
    spin_z[l] = spin_Jz(S[l])
    spin[2] = spin[2]+ tensor(spin_z)
    return spin



def group_by_tolerance_with_indices(values, relative_tolerance):
    """
    Group values by tolerance and return indices for each group.
    """
    groups = []
    group_indices = []  # To store the indices for each group
    used = np.zeros(len(values), dtype=bool)  # Track which values are grouped

    for i, val in enumerate(values):
        if not used[i]:
            # Find all values within tolerance
            group = [
                j for j, v in enumerate(values) 
                if not used[j] and np.allclose(v, val, atol=relative_tolerance)
                ]
            groups.append(values[group])  # Grouped values
            group_indices.append(group)  # Indices of this group
            used[group] = True

    return groups, group_indices



def kondo_rates(J, V_bias, eta, T, eig, states, l, N, S):
    n_states = len(eig)
    spin = spin_operators(l, N, S)
    kondo_scat = sum(tensor(spin[m], sigmas[m]) for m in range(3))

    transition_sums = np.zeros((n_states, n_states), dtype='complex')
    for i in range(n_states):
        for j in range(n_states):
            for bra, ket in itertools.product(range(2), repeat=2):
                pol = (1 + eta[0] * (-1 + 2 * bra)) * (1 + eta[1] * (-1 + 2 * ket))
                bra_vec = tensor(states[i], basis(2, bra))
                ket_vec = tensor(states[j], basis(2, ket))
                transition_sums[i, j] += pol * J[0] * J[1] * np.abs(kondo_scat.matrix_element(bra_vec, ket_vec))**2

    eps_ij = eig[:, None] - eig + V_bias * e
    factors = np.where(
        np.abs(eps_ij) > 1e-10,
        eps_ij / np.expm1(eps_ij / (k_B * T)),
        k_B * T
    )
    return transition_sums * factors * 2 * np.pi

def solve_rate_equations(k, y0, t_list):
    """ solve rate_equations
    Parameters:
    ---------------
    k: nd.array
        rates
    y0: nd.array
        initial populations in energy basis
    t_list: ndarray
        timesteps
    
    Returns:
    y: nd.array
        populations as a function of time
    """
    k_diff = k - np.diag(np.sum(k, axis=0))
    return solve_linear_set_equations(k_diff, y0, t_list)

def solve_rate_phase_equations(k, rho_0, t_list, eig):
    """ solve rate_equations
    Parameters:
    ---------------
    k: nd.array
        rates
    rho_0: nd.array
        initial populations in energy basis
    t_list: ndarray
        timesteps
    eig: ndarray
       eigenvalues of spin Hamiltonian
    
    Returns:
    y: nd.array
        populations as a function of time
    """
    n = np.shape(k)[0]
    nsteps = len(t_list)

    # Compute phase and decoherence matrices
    phase = 1j * (eig[:, None] - eig) * 2 * np.pi
    dec = 0.5 * (np.sum(k, axis=0)[:, None] + np.sum(k, axis=0)[None, :])
    np.fill_diagonal(dec, 0)

    # Analytical solution for coherence terms
    exp_terms = np.exp((-phase - dec)[None, :, :] * t_list[:, None, None])
    coherence_terms = (rho_0 - np.diag(np.diag(rho_0)))[None, :, :] * exp_terms

    # Solve populations via rate equations
    populations = solve_rate_equations(k, np.diag(rho_0).reshape(-1, 1), t_list)

    rho = np.zeros((nsteps, n, n), dtype='complex')
    rho += coherence_terms
    for i in range(nsteps):
        np.fill_diagonal(rho[i], populations[i, :, 0])

    return rho

def solve_linear_set_equations(k, y0, t_list):
    """ Function to solve linear set of first order differential equations by determining eigenvalues and eigenstates
    -------------------
    Parameters:
        k: ndarray
            rate equations in matrix form
        y0: ndarray
            initial state vector
        t_list: ndarray
            Times for the set of differential equations is solved [s] 
    Returns:
        y: State vector for every time step for which the set of equantions is solved
    """

    n_steps = np.size(t_list)
    
    # Initialse the density matrix vector
    y = np.zeros(shape=(int(n_steps), np.shape(y0)[0], 1), dtype=complex)

    # Set up set of differential equations
    e, v = linalg.eig(k)  # Find eigenvalues and eigenvectors
    vec = linalg.solve(v, np.eye(np.shape(e)[0])) @ y0  # Solve for initial conditions
    e = e[:, np.newaxis]

    # Solve set of differential equations
    for j in range(0, n_steps):
        y[j, :] = np.dot(v, np.multiply(vec, np.exp(e * t_list[j])))  # Solution
    return y

def complex_to_rgb(magnitude, phase):
    """
    Custom function to map complex values (magnitude + phase) to RGB colors.
    
    - Phase (angle) determines the hue.
    - Magnitude controls brightness (0 = white, 1 = full color).
    """
    # Normalize phase from [-π, π] to [0, 1]
    norm_phase = (phase + np.pi) / (2 * np.pi)  
    
    # Define custom phase-to-color mapping
    color_map = np.array([
        [1, 0, 0],   # Red for 0° (real positive, phase = 0)
        [1, 0.6, 0.8],  # Pink for 90° (imaginary positive, phase = π/2)
        [0, 1, 0],   # Green for 180° (real negative, phase = π)
        [0, 0.5, 1],   # Marine blue for -90° (imaginary negative, phase = -π/2)
        [1, 0, 0]    # Red again to close the loop (360°)
    ])

    # Interpolate between these colors based on phase
    phase_idx = norm_phase * (len(color_map) - 1)  
    lower_idx = np.floor(phase_idx).astype(int)
    upper_idx = np.ceil(phase_idx).astype(int) % len(color_map)
    blend_factor = phase_idx - lower_idx
    
    # Linearly interpolate between two nearest colors
    rgb = (1 - blend_factor) * color_map[lower_idx] + blend_factor * color_map[upper_idx]

    # Adjust brightness by magnitude (fade to white at low magnitudes)
    rgb = (1 - magnitude) + magnitude * rgb  # Blend with white
    return np.clip(rgb, 0, 1)  # Ensure valid RGB values
