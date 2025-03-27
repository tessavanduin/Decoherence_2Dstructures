# Author: Tessa van Duin
# Based on code by: Rik Broekhoven
#
# Date: 26/03/25


import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from matplotlib import colormaps
from matplotlib.widgets import Slider
import matplotlib.ticker as mticker
from qutip import Qobj, tensor, spin_Jm, spin_Jz, basis, sigmay, fidelity
from qutip import fidelity, tracedist
from scipy import linalg
from itertools import combinations
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from joblib import Parallel, delayed
import imageio
import os
import pickle
import seaborn as sns
from Back_end_ClassManager import hamiltonian, complex_to_rgb, group_by_tolerance_with_indices, simple_square_lattice, simple_hexagonal_lattice, h_ident, kondo_rates, solve_rate_phase_equations

h = 6.62607015e-34  # Planck constant [Js]
hbar = h / (2 * np.pi)  # Reduced Planck constant [Js/rad]
k_B = 1.380649e-23  # Boltzmann constant in J/K (you may set to 1 in natural units)

class EnergySpectrumManager:
    def __init__(self, structure, gyro, N, S=[1/2], B=0, Btip=0, rates_file='rates.pkl'):
        """
        Initialize the manager with system parameters.
        
        Parameters:
        -----------
        structure: ndarray
            The atomic structure or interaction structure matrix.
        gyro: ndarray
            Gyromagnetic ratios for each spin.
        J: ndarray
            Exchange coupling constants.
        D: ndarray
            Anisotropic coupling constants.
        N: int
            Number of spins.
        S: list, optional
            Spin quantum numbers (default: [1/2]).
        Btip: float, optional
            Tip magnetic field for the first spin (default: 0).

        """
        self.structure = structure
        self.gyro = gyro
        self.N = N
        if N is None:
            N = np.shape(structure)        
        self.S = S
        self.B = B
        self.Btip = Btip
        self.highlight_states = []  # List to store eigenstates to highlight
        self.colors = colormaps["tab10"] # Colors for highlighted states
        self.saved_rates = None  # Initialize saved rates
        self.rates_file = rates_file
        self.load_rates()  # Load previously saved rates if they exist
    
    def interaction_strengths(self, B=0.9, J0=[27.7e9, 0.72e-9], a=2.88e-10, dex=94e-12, mu0 = 1.25663706212e-6):
        """
        Calculation of interaction strengths (J and D) between spins.

        """
        # Ensure self.structure is a NumPy array
        self.structure = np.asarray(self.structure)
        
        # Compute atom combinations and distances
        atomcomb = np.array(list(combinations(range(self.N), 2)))
        distancematrix = self.structure[atomcomb[:, 0]] - self.structure[atomcomb[:, 1]]
        rdim = linalg.norm(distancematrix, axis=1)
        r = rdim * a  # Distances scaled by lattice constant

        # Exchange coupling calculation
        J = J0[0] * np.exp(-(r - J0[1]) / dex) / 1e9

        # Dipolar coupling calculation
        gyroproduct = np.multiply(self.gyro[atomcomb[:, 0]], self.gyro[atomcomb[:, 1]])
        D0 = mu0 * gyroproduct * h / (r**3 * 16 * np.pi**3 * 1e9)

        # Store the computed values in the object
        self.J = J
        self.D = D0
        self.atomcomb = atomcomb

    def compute_eigenvalues_and_states(self, B):
        """
        Compute eigenvalues and eigenstates for a given magnetic field.
        
        """
        # Compute Larmor frequencies
        larmor = np.array(self.gyro[:, np.newaxis] * B / (2 * np.pi * 1e9))
        larmor[0, :] += self.gyro[0] * self.Btip / (2 * np.pi * 1e9)

        # Build Hamiltonian
        self.interaction_strengths(B=B, a=2.88e-10, dex=94e-12, mu0 = 1.25663706212e-6)
        # self.D = np.zeros(6)
        # self.J = np.zeros(6)
        H = hamiltonian(lattice=self.structure, larmor=larmor, J=self.J, D=self.D, N=self.N, S=self.S)
        
        eig, states = H.eigenstates()
        eig /= 2 * np.pi  # Convert to frequency (GHz)
        
        self.eig = eig
        self.states = states

    def get_spin_state_representation(self, states, B, index, tol=1e-6):
        """
        Returns the spin basis state decomposition of the eigenstate corresponding to a given eigenvalue.
        
        Parameters:
            eigenvalues (numpy array): Array of eigenvalues of the system.
            eigenstates (numpy array): 2D array of eigenstates (columns correspond to eigenvectors).
            target_eigenvalue (float): The eigenvalue for which we want the corresponding eigenstate.
            tol (float): Tolerance for matching eigenvalues (to account for numerical precision).
            
        Returns:
            str: String representation of the eigenstate in spin basis notation.
        """
        self.compute_eigenvalues_and_states(B=B)

        # Get the corresponding eigenstate
        print(self.eig[index])
        state_vector = states[index]

        # Generate spin basis labels
        num_spins = int(np.log2(state_vector.shape[0]))  # Determine the number of spins from the length
        spin_basis = [format(i, f'0{num_spins}b') for i in range(state_vector.shape[0])]  # Binary representations

        # Construct the state representation
        state_representation = []
        for coeff, basis in zip(state_vector, spin_basis):
            coeff = coeff.item()  # Convert from numpy array to scalar

            if np.abs(coeff) > tol:  # Ignore very small contributions
                # Convert binary to ket notation
                ket = "".join("↑" if b == "1" else "↓" for b in basis)
                # Format the coefficient
                coeff_str = f"{coeff.real:.3f}" if np.isclose(coeff.imag, 0, atol=tol) else f"({coeff.real:.3f} + {coeff.imag:.3f}j)"
                state_representation.append(f"{coeff_str} * |{ket}⟩")

        # Join the terms into a single string
        return " + ".join(state_representation) if state_representation else "0"

    def plot_eigenstate_complex_contributions(self, d, B):
        """
        Plots spin eigenstate contributions with a complex-valued colormap.
        The color legend (complex circle) is now an inset inside the main figure.
        """
        self.compute_eigenvalues_and_states(B=B)

        eigenvectors = np.array([state.full().flatten() for state in self.states])
        basis_states = np.eye(len(eigenvectors))  
        overlaps_complex = eigenvectors @ basis_states.T  
        # print(overlaps_complex)
        for i in range(4):
            print(f'eigenvalue of {i} is {self.eig[i]}')
            print(f'eigenstate of {i} is {self.states[i]}')

        magnitude = np.abs(overlaps_complex)
        phase = np.angle(overlaps_complex)

        magnitude_norm = magnitude / magnitude.max() if magnitude.max() > 0 else magnitude
        rgb_colors = np.array([complex_to_rgb(mag, ph) for mag, ph in zip(magnitude_norm.flatten(), phase.flatten())])
        rgb_colors = rgb_colors.reshape(*magnitude_norm.shape, 3)  

        rgb_colors_flipped = rgb_colors[::-1]
        eigenvalues_flipped = np.round(self.eig, 2)[::-1]

        spin_states = [''.join(map(str, s)) for s in itertools.product([0, 1], repeat=self.N)]
        arrow_labels = [state.replace("1", "↑").replace("0", "↓") for state in spin_states]

        fig, axes = plt.subplots(1, 2, figsize=(16, 9.12), gridspec_kw={'width_ratios': [3, 1]})

        # 🔹 Plot heatmap (left)
        ax_main = axes[0]
        ax_main.imshow(rgb_colors_flipped, aspect="auto")

        # ax_main.set_xlabel("Spin Basis States", fontsize=28)
        # ax_main.set_ylabel("Eigenvalues (GHz)", fontsize=28)
        ax_main.set_xticks(np.arange(len(arrow_labels)))
        ax_main.set_xticklabels(arrow_labels, fontsize=28, rotation=45, ha="right")
        ax_main.set_yticks(np.arange(len(self.eig)))
        ax_main.set_yticklabels(eigenvalues_flipped, fontsize=28)


        # ax_main.set_title("Eigenstate Contributions (Complex Colormap)", fontsize=18)

        # 🔹 Plot color circle legend (right)
        ax_circle = axes[1]
        size = 300
        x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
        r = np.sqrt(x**2 + y**2)  
        theta = np.arctan2(y, x)  

        mask = r > 1
        r[mask] = 0
        theta[mask] = 0

        rgb_circle = np.array([complex_to_rgb(m, p) for m, p in zip(r.flatten(), theta.flatten())])
        rgb_circle = rgb_circle.reshape(*r.shape, 3)  

        ax_circle.imshow(rgb_circle, extent=[-1, 1, -1, 1], origin="lower")

        # Add real and imaginary axes with labels
        ax_circle.axhline(0, color='black', linestyle='--', linewidth=1)  # Real axis
        ax_circle.axvline(0, color='black', linestyle='--', linewidth=1)  # Imaginary axis

        ax_circle.text(1.05, 0, "1", fontsize=22, verticalalignment='center', horizontalalignment='left', color='black')
        ax_circle.text(-1.05, 0, "-1", fontsize=22, verticalalignment='center', horizontalalignment='right', color='black')
        ax_circle.text(0, 1.05, "i", fontsize=22, horizontalalignment='center', verticalalignment='bottom', color='black')
        ax_circle.text(0, -1.05, "-i", fontsize=22, horizontalalignment='center', verticalalignment='top', color='black')
        # Labels for axes
        ax_circle.text(1.45, 0, "Re", fontsize=28, verticalalignment='center', color='black')
        ax_circle.text(0, 1.45, "Im", fontsize=28, horizontalalignment='center', color='black')

        ax_circle.set_xticks([])
        ax_circle.set_yticks([])
        ax_circle.set_frame_on(False)  # Removes the square border
        # ax_circle.set_title("Complex Phase & Magnitude", fontsize=14)

        plt.tight_layout()
        filename = f'triangle_dist{d}_matrix_{B[2]}T.pdf'
        plt.savefig(filename)
        # plt.show()

    def plot_eigenstate_contributions(self, d, B):
        """
        Plots the contributions of spin basis states to each eigenvalue as a heatmap.
        """
        self.compute_eigenvalues_and_states(B=B)

        for i in range(len(self.states)):
            print(f'eigenvalue of {i} is {self.eig[i]}')
            print(f'eigenstate of {i} is {self.states[i]}')

        # Convert eigenstates to NumPy arrays
        eigenvectors = np.array([state.full().flatten() for state in self.states])  # Shape (num_states, dim)

        # Generate computational basis states (Identity matrix in Hilbert space)
        basis_states = np.eye(len(eigenvectors))  # Computational basis in standard basis

        # Compute overlap probabilities |⟨basis | eigenvector⟩|²
        # overlaps = np.abs(eigenvectors @ basis_states.T) ** 2  
        overlaps = eigenvectors @ basis_states.T
        overlaps_flipped = overlaps[::-1]
        eigenvalues_flipped = np.round(self.eig, 2)[::-1]

        spin_states = [''.join(map(str, s)) for s in itertools.product([0, 1], repeat=self.N)]

        # Convert binary states to arrow representation (1 → ↑, 0 → ↓)
        arrow_labels = [state.replace("1", "↑").replace("0", "↓") for state in spin_states]
        
        # Plot heatmap
        plt.figure(figsize=(10, 7))
        plt.imshow(overlaps_flipped, cmap="coolwarm", aspect="auto")
        # Add colorbar and change fontsize
        im = plt.imshow(overlaps, cmap="coolwarm", aspect="auto")
        cbar = plt.colorbar(im)
        cbar.ax.tick_params(labelsize=18)

        plt.xlabel("Spin Basis States", fontsize=18)
        plt.ylabel("Eigenvalues (GHz)", fontsize=18)
        plt.xticks(ticks=np.arange(len(arrow_labels)), labels=arrow_labels,fontsize=18, rotation=45, ha="right")
        plt.yticks(ticks=np.arange(len(self.eig)), labels=eigenvalues_flipped, fontsize=18)
        plt.tight_layout()
        # plt.title("Eigenstate Contributions")
        plt.show()
        # filename = f'square_2x2_dist{d}_matrix.png'
        # plt.savefig(filename)

    def initialize_highlight_states(self, num_highlight=0):
        """
        Initialize the highlighted eigenstates from the lowest eigenvalues.
        
        """
        # Identify the indices of the lowest eigenvalues
        sorted_indices = np.argsort(self.eig)
        if num_highlight != 0:
            lowest = sorted_indices[:num_highlight]
            highest = sorted_indices[-num_highlight:]
            selected_indices = np.concatenate((lowest, highest))
        else:
            selected_indices = []
        # Store the corresponding eigenstates
        self.highlight_states = [self.states[i] for i in selected_indices]

    def update_highlight_states(self):
        """
        Update the highlighted eigenstates by matching overlaps with stored states.
        
        """
        new_highlight_states = []
        for hs in self.highlight_states:
            # Compute overlaps with current states
            overlaps = np.array([abs(state.overlap(hs)) for state in self.states])
            # Find the index of the state with the maximum overlap
            max_overlap_idx = np.argmax(overlaps)
            # Append the matched state
            new_highlight_states.append(self.states[max_overlap_idx])
        self.highlight_states = new_highlight_states

    def plot_energy_spectrum(self, return_data=False, B=0, save_path=None, tolerance=1e-5):
        """
        Plot the energy spectrum for a given magnetic field, highlighting the tracked eigenstates.
        Optionally, return the data for further processing or save the plot to a file.
        """

        if return_data:
            data = {'x': [], 'y': [], 'highlight_x': [], 'highlight_y': [], 'highlight_colors': []}

        x_vals = np.arange(len(self.eig))
        if return_data:
            data['x'] = x_vals
            data['y'] = self.eig

        # Initialize lists for highlights
        highlight_x = []
        highlight_y = []
        highlight_color = []

        # Highlight tracked eigenstates
        for idx, hs in enumerate(self.highlight_states):
            overlaps = np.abs([state.overlap(hs) for state in self.states])
            max_overlap_idx = np.argmax(overlaps)

            highlight_x.append(max_overlap_idx)
            highlight_y.append(self.eig[max_overlap_idx])
            highlight_color.append(self.colors[idx % len(self.colors)])

            if return_data:
                data['highlight_x'] = highlight_x
                data['highlight_y'] = highlight_y
                data['highlight_colors'] = highlight_color

        if not return_data:
            fig, ax = plt.subplots(figsize=(16, 9.12))

            # Assign colors to degenerate eigenvalues
            eigenvalue_colors = ['lightgray'] * len(self.eig)
            color_palette = colormaps['tab10'].colors  # Assuming self.colors is predefined
            color_index = 0
            for i, eig_i in enumerate(self.eig):
                if eigenvalue_colors[i] == 'lightgray':  # Only assign color if not yet assigned
                    # Check for degeneracies with relative tolerance
                    degenerate_indices = [  
                        j for j in range(i, len(self.eig)) 
                        if abs(self.eig[j] - eig_i) < tolerance * abs(eig_i)  # Relative tolerance
                    ]

                    if len(degenerate_indices) > 1:  # Degenerate group found
                        # Assign a new color to all members of the degenerate group
                        current_color = color_palette[color_index % len(color_palette)]
                        for j in degenerate_indices:
                            eigenvalue_colors[j] = current_color
                        color_index += 1


            # Plot the eigenvalues
            for i, eig_i in enumerate(self.eig):
                print(eig_i)
                ax.plot(x_vals[i], eig_i, 'o', markersize=12, color=eigenvalue_colors[i])

            # Overlay highlights
            for x, y, color in zip(highlight_x, highlight_y, highlight_color):
                ax.plot(x, y, 'o', markersize=12, color=color)

            ax.set_xscale("linear")
            # ax.set_yscale("symlog", linthresh=0.01)
            ax.set_xlabel("Eigenenergy index", fontsize=28)
            ax.set_ylabel("Eigenenergy (GHz)", fontsize=28)

            yticks = plt.gca().get_yticks()
            # ax.set_title(f"Hamiltonian Energy Spectrum at Magnetic Field B ={B[2]}", fontsize=24)
            # ax.legend()
            # Increase the font size of the tick labels
            ax.tick_params(axis='both', which='major', labelsize=28)  # Major ticks
            ax.tick_params(axis='both', which='minor', labelsize=24)  # Minor ticks

            ax.grid(True, which="both", linestyle="--", linewidth=0.5)
            plt.tight_layout()

            if save_path:  # Save the plot if save_path is provided
                plt.savefig(save_path)
                plt.close(fig)  # Close the figure after saving
            else:  # Show the plot otherwise
                plt.show()
        else:
            return data

    def run_and_plot(self, B, tolerance, d):
        self.compute_eigenvalues_and_states(B)
        self.plot_energy_spectrum(return_data=False,  B=B, tolerance=tolerance, save_path=f'triangle_dist{d}_{B[2]}T.pdf')
    
    def run_and_plot_distance(self, R_values, B):
        """
        Compute and plot eigenvalues as a function of interatomic distance.
        
        """
        self.B = B
        eigenvalues = []  # Store eigenvalues for each distance

        for r in R_values:
            # Update the lattice structure
            structure = simple_hexagonal_lattice(1, r, True)
            self.update_structure(structure)

            # Compute eigenvalues for the updated structure
            self.compute_eigenvalues_and_states(self.B)  # Use a predefined magnetic field
            eigenvalues.append(self.eig)

        # Convert eigenvalues to a numpy array for plotting
        eigenvalues = np.array(eigenvalues).T

        # Plot eigenvalues as a function of distance
        plt.figure(figsize=(10, 6))
        for eig in eigenvalues:
            plt.plot(R_values, eig, '-o', markersize=4)

        plt.yscale("symlog")
        plt.xlabel("Interatomic Distance (r)")
        plt.ylabel("Eigenenergy (GHz)")
        plt.title("Eigenvalue Evolution with Interatomic Distance")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()





    def plot_interaction_decay(self, R_values, shape='Square', combinationindex=None, a=2.88e-10, J0=27.7e9, dex=94e-12, mu0=1.25663706212e-6, tolerance=1e-2):
        """
        Plot the decay of exchange coupling (J) and dipolar coupling (D) over distance,
        grouping J and D values that are within a specified tolerance.
        """
        # Initialize lists to store results
        J_values = []
        D_values = []
        r_distances = []

        col = None
        row = None
        middle = None
        for r in R_values:
            # Update structure with the current distance
            if shape == 'Hex':
                if middle is None:
                    middle = str(input("Middle included in hexagon (True/False): "))
                self.structure = simple_hexagonal_lattice(1, r, middle=middle)  # Example lattice method
            elif shape == 'Square':
                if col is None or row is None:
                    col = int(input("Provide number of columns: "))
                    row = int(input("Provide number of rows: "))
                self.structure = simple_square_lattice(col, row, r)
                self.N = np.shape(self.structure)[0]
                g = np.array([1.74]*self.N)   
                muB = 9.27400915e-24  # Bohr magnetorn [J/T]
                self.gyro = (g * muB) / hbar  # Example gyromagnetic ratios for 4 spins
            
            # Compute interaction strengths
            self.interaction_strengths()

            # Append values for plotting
            J_values.append(self.J)
            D_values.append(self.D)
            r_distances.append(linalg.norm(self.structure[self.atomcomb[:, 0]] - self.structure[self.atomcomb[:, 1]], axis=1) * a)

        # Convert to arrays
        J_values = np.array(J_values).T
        D_values = np.array(D_values).T
        r_distances = np.array(r_distances).T

        J_groups, group_indices = group_by_tolerance_with_indices(J_values, tolerance)
        D_groups = [D_values[indices] for indices in group_indices]

        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 7))

        # Set up colors for each group using a colormap
        cmap = colormaps["tab10"]  # Get the 'tab10' colormap
        numcol = max(len(J_groups), len(D_groups))
        colors = cmap(np.linspace(0, 1, numcol))  # Generate distinct colors for groups

        # Plot grouped Exchange Coupling (J)
        for group_id, indices in enumerate(group_indices):
            label = [u'Vertical, Δx=0 / Horizontal, Δy=0', 'Diagonal, x=y']
            group_average_J = np.mean(J_values[indices], axis=0)
            ax.plot(2*R_values, group_average_J, '-o', color=colors[group_id], 
                    label=f'Exchange pair {group_id+1}') # {label[group_id]}') #

        # Plot grouped Dipolar Coupling (D) using the same indices
        for group_id, indices in enumerate(group_indices):
            label = [u'Vertical, Δx=0 / Horizontal, Δy=0', 'Diagonal, x=y']
            group_average_D = np.mean(D_values[indices], axis=0)
            ax.plot(2*R_values, group_average_D, '--s', color=colors[group_id], 
                    label=f'Dipolar pair {group_id+1}') #{label[group_id]}') #

        # Set log scale for both axes
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%d'))
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.set_xlabel("Interatomic Distance (r)", fontsize=16)
        ax.set_ylabel("Exchange (J) and Dipolar (D) Coupling (GHz)", fontsize=16)
        ax.set_title("Coupling Decay with Interatomic Distance", fontsize=16)
        ax.legend(fontsize=14, loc=6)
        ax.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    def load_rates(self):
        """Load saved rates from file if available."""
        try:
            with open(self.rates_file, 'rb') as f:
                self.saved_rates = pickle.load(f)
        except FileNotFoundError:
            self.saved_rates = None  # If no saved rates, compute them later

    def save_rates(self):
        """Save computed rates to a file."""
        with open(self.rates_file, 'wb') as f:
            pickle.dump(self.saved_rates, f)

    def thermal_state(self, T):
        beta = 1 / (hbar * k_B * T)        # hbar

        # Shift energies to avoid overflow: E' = E - E_min
        E_min = np.min(self.eig)
        shifted_eig = self.eig - E_min  

        # Compute Boltzmann weights safely
        exp_eig = np.exp(-beta * np.array(shifted_eig))  

        # Normalize probabilities
        Z = np.sum(exp_eig)  # Partition function
        exp_eig /= Z  

        # Construct density matrix from eigenvectors
        rho = sum(p * (state * state.dag()) for p, state in zip(exp_eig, self.states))
        return rho

    def structure_decay(self, B, T=1e-2, flip_spin=False, coherence=True, times=np.linspace(0, 1e2, 1000), assymmetric=False, reset=False):
        if reset:
            self.saved_rates = None

        self.nsteps = len(times)

        # Compute eigenvalues and states using the existing 2D Hamiltonian
        self.compute_eigenvalues_and_states(B)

        # Ground state of the system
        # op_p = h_ident(self.N, self.S)
        # op_p[flip_spin] = spin_Jm(self.S[flip_spin])
        # psi_0 = tensor(op_p) @ self.states[0]
        # rho_ground = psi_0 * psi_0.dag()
        
        # Thermal equilibrium initialisation
        rho_0 = self.thermal_state(T=T)
        rho_thermal = self.thermal_state(T=T)  # Get the thermal equilibrium state
        rho_thermal /= rho_thermal.tr()
        op_p = h_ident(self.N, self.S)  # Identity operators for all sites
        op_p[0] = spin_Jm(self.S[0])    # Lowering operator on spin 0
        op_full = tensor(op_p)  # Expand to the full system
        # Apply lowering operator to the thermal state
        rho_0 = op_full * rho_thermal * op_full.dag()
        rho_0 /= rho_0.tr()  # Normalize

        # Precompute e_ops
        e_ops = [
            tensor([
                spin_Jz(self.S[l]) if l == idx else h_ident(self.N, self.S)[idx] for idx in range(self.N)
            ]) for l in range(self.N)
        ]

        if self.saved_rates is None:
            print("Recalculating rates...")
            J_surf = np.full(self.N, 8e-3)
            if assymmetric:
                J_surf[0] = 0
            # Parallel computation of rates
            rate_tot = sum(
                Parallel(n_jobs=-1)(
                    delayed(kondo_rates)(
                        J=[J_surf[l], J_surf[l]],
                        V_bias=0,
                        eta=[0, 0],
                        T=T,
                        eig=self.eig,
                        states=self.states,
                        l=l,
                        N=self.N,
                        S=self.S
                    ) for l in range(self.N)
                )
            )
            self.saved_rates = rate_tot
            self.save_rates()
        else:
            print("Using saved rates.")
            rate_tot = self.saved_rates

        # Solve rate equations and compute density matrix evolution
        rho = solve_rate_phase_equations(rate_tot, rho_0.transform(self.states).full(), times, self.eig)
        if coherence:
            return self.states, rho

        # Compute expectation values
        result = np.array([
            [
                np.trace(Qobj(rho[j]).transform(self.states, inverse=True).full() @ e_ops[i].full())
                for j in range(len(times))
            ] for i in range(self.N)
        ])
        return result

    def plot_structure_decay(self, B, dist, T=0.5, flip_spin=False, coherence=False, times=np.linspace(0, 1e2, 1000), assymmetric=False, reset=False):
        result = self.structure_decay(B=B, T=T, flip_spin=False, coherence=coherence, times=times, assymmetric=assymmetric, reset=reset)

        # If coherence is False, result is likely a tuple with (states, rho), so get the second element
        if isinstance(result, tuple):
            result = result[1]  # Assuming the second part is the decay result we need for plotting
     
        ax = plt.subplots()[1]
        cmap = colormaps["tab10"]  # Get the 'tab10' colormap
        colors = cmap(np.linspace(0, 1, self.N))

        for i in range(self.N):
            ax.plot(times, np.real(result[i]), color=colors[i % len(colors)], label=f'n={i}')
            # print(f'The starting position of atom {i} is: {result[i][0]}')
            # print(f'The ending position of atom {i} is: {result[i][-1]}')
        
        ax.set_ylabel(r'$\langle S_z \rangle$', fontsize=16)
        ax.set_xlabel('time (ns)', fontsize=16)

        ax.tick_params(axis='both', which='major', labelsize=14)  # Major ticks
        ax.tick_params(axis='both', which='minor', labelsize=12)  # Minor ticks
        ax.set_ylim([-.55, .55])
        ax.legend(fontsize=14, loc=4)
        # Remove axes, ticks, and labels
        # ax.axis("off")  # Hide axes completely
        fig_name = f'Square_pres_{dist}_{B[2]}T.png'
        plt.savefig(fig_name, bbox_inches='tight')
        plt.show()
        
