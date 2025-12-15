from matplotlib import animation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from chemistry import Chemistry, ChemistryManager
from fluid import Fluid
import constant as const
import os
import time
from numba import njit, prange


class System:
    """
    System controller for the coupled fluid-chemistry simulation.

    This class encapsulates the simulation domain, numerical settings,
    physical fields (velocity, temperature, species mass fractions),
    boundary conditions, time-stepping logic and I/O/visualization helpers.

    Responsibilities:
    - define computational grid and physical geometry (slots, coflow, heated rod)
    - compute stable time step from CFL and Fourier criteria
    - initialize and coordinate Fluid and ChemistryManager components
    - perform a fractional-step time advance (predictor, pressure solve, correct)
    - advance species and temperature (including RK4 substepping for chemistry)
    - save datasets and produce plots/animations
    """

    def __init__(self, dt_data: float, total_time: float, n: int, m: int, fluid: Fluid, ChemicalManager: ChemistryManager):
        """
        Initialize the System instance.

        Args:
            dt_data (float): interval (s) between saved data snapshots.
            total_time (float): total simulation time (s).
            n (int): number of grid points in x-direction.
            m (int): number of grid points in y-direction.
            fluid (Fluid): Fluid object that holds velocity fields and solvers.
            ChemicalManager (ChemistryManager): manager for species, reactions and chemistry solvers.

        What this initializer does:
        - store time and grid parameters and compute dx, dy
        - set physical geometry (domain size, slot/coflow lengths, heated rod)
        - compute a stable fluid time step from CFL and diffusion (Fourier) constraints
        - attach and initialize the Fluid and ChemistryManager with grid and time info
        - initialize temperature field with a centered heated rod and default inlet conditions
        """
        # === Time parameters ===
        self.dt_data = dt_data
        self.total_time = total_time
        self.current_time = 0.0

        # === Grid dimensions ===
        self.n = n
        self.m = m

        # === Domain physical dimensions (m) ===
        self.Lx = 2e-3              # Domain length in x
        self.Ly = 2e-3              # Domain width in y
        self.Lslot = 0.5e-3         # Length of fuel/oxidizer slot
        self.Lcoflow = 0.5e-3       # Length of coflow region
    
        # === Spatial discretization ===
        self.dx = self.Lx / (self.n - 1)
        self.dy = self.Ly / (self.m - 1)
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy

        # === Inlet boundary conditions ===
        self.Uslot = 1.0            # Slot inlet velocity (m/s)
        self.Tslot = 300.0          # Slot inlet temperature (K)
        self.Ucoflow = 0.2          # Coflow inlet velocity (m/s)
        self.Tcoflow = 300.0        # Coflow inlet temperature (K)

        # === Index markers for different boundary regions ===
        self.ind_inlet = int(self.Lslot / self.dx)                     # End of inlet slot
        self.ind_coflow = int((self.Lslot + self.Lcoflow) / self.dx)   # End of coflow region
        
        # === Time step calculation based on stability criteria ===
        # CFL condition for advection: Δt ≤ CFL * Δx / U
        # Fourier condition for diffusion: Δt ≤ Fo * Δx² / ν
        self.CFL = 0.25
        self.Fo = 0.15
        dt_cfl = self.CFL * self.dx / self.Uslot
        dt_fourier = self.Fo * self.dx**2 / const.nu
        self.dt = np.min((dt_cfl, dt_fourier))

        # === Physical fields ===
        self.fluid = fluid
        self.ChemicalManager = ChemicalManager

        self.fluid.initiate_steps(self.dt, self.dx, self.dy)
        self.ChemicalManager.initiate_steps(self.dx, self.dy)

        # === Temperature field initialization === Circular heated rod in center

        self.rode = 5e-4 # Width of the heating rode (m)
        self.T_rode = 1000.0  # Temperature of the heated rod (K)
        self.T = np.ones((self.n, self.m)) * 300        # Temperature field (K) with heated rod of size self.rode centered in domain
        x = np.linspace(0.0, self.Lx, self.n)
        y = np.linspace(0.0, self.Ly, self.m)
        X, Y = np.meshgrid(x, y, indexing='ij')
        mask = np.abs(Y - self.Ly/2) <= self.rode / 3
        self.T[mask] = self.T_rode
    
    
    # def step_vectorized_interior(self, u, v, field, inv_dx, inv_dy, diffusion_coef, source=None, chemical_dt = None):
    #     # === Define interior region (excluding boundaries) ===
    #     i_slice = slice(1, -1)
    #     j_slice = slice(1, -1)
        
    #     # Extract local field values
    #     u_loc = u[i_slice, j_slice]
    #     v_loc = v[i_slice, j_slice]
    #     field_loc = field[i_slice, j_slice]
        
    #     # === Upwind advection in x-direction ===
    #     # Use backward difference when u > 0, forward when u < 0
    #     u_pos = np.maximum(u_loc, 0)
    #     u_neg = np.minimum(u_loc, 0)
    #     adv_x = (u_pos * (field_loc - field[:-2, j_slice]) * inv_dx + 
    #              u_neg * (field[2:, j_slice] - field_loc) * inv_dx)
        
    #     # === Upwind advection in y-direction ===
    #     v_pos = np.maximum(v_loc, 0)
    #     v_neg = np.minimum(v_loc, 0)
    #     adv_y = (v_pos * (field_loc - field[i_slice, :-2]) * inv_dy + 
    #              v_neg * (field[i_slice, 2:] - field_loc) * inv_dy)
        
    #     # === Diffusion using central differences (5-point stencil) ===
    #     diff_x = (field[2:, j_slice] - 2*field_loc + field[:-2, j_slice]) * inv_dx**2
    #     diff_y = (field[i_slice, 2:] - 2*field_loc + field[i_slice, :-2]) * inv_dy**2
    #     diff = diffusion_coef * (diff_x + diff_y)
        
    #     # === Explicit time integration: φ^(n+1) = φ^n + Δt * RHS ===
    #     if source is not None:
    #         if chemical_dt is not None:
    #             return field_loc + chemical_dt * (-adv_x - adv_y + diff + source[i_slice, j_slice])
    #         else:
    #             return field_loc + self.dt * (-adv_x - adv_y + diff + source[i_slice, j_slice])
    #     else:
    #         if chemical_dt is not None:
    #             return field_loc + chemical_dt * (-adv_x - adv_y + diff)
    #         else:
    #             return field_loc + self.dt * (-adv_x - adv_y + diff)

    def temperature_boundary(self, T_new):
        # --- Left boundary (i=0) - Neumann (zero gradient) ---
        i = 1
        for j in range(2, self.m-2):
            T_new[i, j] = T_new[i+1, j]
        T_new[0, 1:-2] = T_new[1, 1:-2]
        
        # --- CH4 inlet (slot region, bottom wall j=0) ---
        T_new[:self.ind_inlet, 0] = self.Tslot
        T_new[:self.ind_inlet, 1] = self.Tslot

        # --- O2+N2 inlet (slot region, top wall j=m-1) ---
        T_new[:self.ind_inlet, self.m-1] = self.Tslot
        T_new[:self.ind_inlet, self.m-2] = self.Tslot

        # --- N2 coflow inlet (coflow region, bottom wall j=0) ---
        T_new[self.ind_inlet:self.ind_coflow, 0] = self.Tcoflow
        T_new[self.ind_inlet:self.ind_coflow, 1] = self.Tcoflow

        # --- N2 coflow inlet (coflow region, top wall j=m-1) ---
        T_new[self.ind_inlet:self.ind_coflow, self.m-1] = self.Tcoflow
        T_new[self.ind_inlet:self.ind_coflow, self.m-2] = self.Tcoflow

        # --- Lower wall (outlet region, j=0) - Neumann ---
        T_new[self.ind_coflow:, 1] = T_new[self.ind_coflow:, 2]
        T_new[self.ind_coflow:, 0] = T_new[self.ind_coflow:, 2]

        # --- Upper wall (outlet region, j=m-1) - Neumann ---
        T_new[self.ind_coflow:, self.m-2] = T_new[self.ind_coflow:, self.m-3]
        T_new[self.ind_coflow:, self.m-1] = T_new[self.ind_coflow:, self.m-3]

        # --- Right boundary (outlet, i=n-1) - Extrapolation ---
        T_new[self.n-2, 2:self.m-2] = T_new[self.n-3, 2:self.m-2]
        T_new[self.n-1, 2:self.m-2] = T_new[self.n-3, 2:self.m-2]

    # def update_temperature(self, u_new, v_new, heat, dt, i_int = slice(1, -1), j_int = slice(1, -1)):
    #     T_new = np.copy(self.T)
    #     # --- Temperature field ---
    #     T_source = heat / (const.rho * const.c_p)
    #     T_interior = self.step_vectorized_interior(
    #         u_new, v_new, self.T, 
    #         1/self.dx, 1/self.dy, const.nu, T_source
    #     )
        
    #     # === UPDATE INTERIOR TEMPERATURE === 
    #     T_new[i_int, j_int] = T_interior
    #     if self.current_time <= self.Lx / self.Uslot:
    #         x = np.linspace(0.0, self.Lx, self.n)
    #         y = np.linspace(0.0, self.Ly, self.m)
    #         X, Y = np.meshgrid(x, y, indexing='ij')
    #         # circular heated rod centered in domain
    #         mask = np.abs(Y - self.Ly/2) <= self.rode / 2
    #         T_new[mask] = self.T_rode
        
    #     # === APPLY SCALAR BOUNDARY CONDITIONS ===
    #     self.temperature_boundary(T_new)
        
    #     self.T = T_new
    
    def compute_species_rhs(self, phi, u, v, diffusion_coef, source):
            # Use interior excluding two layers to allow 4th-order 5-point stencil
        i = slice(2, -2); j = slice(2, -2)
        phi_loc = phi[i, j]
        # shifts in x
        phi_m2_x = phi[:-4, j]
        phi_m1_x = phi[1:-3, j]
        phi_p1_x = phi[3:-1, j]
        phi_p2_x = phi[4:, j]
        # shifts in y
        phi_m2_y = phi[i, :-4]
        phi_m1_y = phi[i, 1:-3]
        phi_p1_y = phi[i, 3:-1]
        phi_p2_y = phi[i, 4:]

        # advective (upwind) using one-cell offsets (kept compatible with interior slice)
        u_loc = u[i, j]; v_loc = v[i, j]
        u_pos = np.maximum(u_loc, 0); u_neg = np.minimum(u_loc, 0)
        adv_x = (u_pos * (phi_loc - phi_m1_x) * self.inv_dx +
                 u_neg * (phi_p1_x - phi_loc) * self.inv_dx)
        v_pos = np.maximum(v_loc, 0); v_neg = np.minimum(v_loc, 0)
        adv_y = (v_pos * (phi_loc - phi_m1_y) * self.inv_dy +
                 v_neg * (phi_p1_y - phi_loc) * self.inv_dy)

        # 4th-order central approximation of second derivative (5-point stencil)
        diff_x = (-phi_p2_x + 16.0*phi_p1_x - 30.0*phi_loc + 16.0*phi_m1_x - phi_m2_x) * self.inv_dx**2 / 12.0
        diff_y = (-phi_p2_y + 16.0*phi_p1_y - 30.0*phi_loc + 16.0*phi_m1_y - phi_m2_y) * self.inv_dy**2 / 12.0

        diff = diffusion_coef * (diff_x + diff_y)
        src = source[i, j] if source is not None else 0.0
        return -adv_x - adv_y + diff + src

    def rk4_advance_temperature(self, u, v, heat, dt):
        T = np.copy(self.T)
        i_int = slice(2, -2); j_int = slice(2, -2)
        T_source = heat / (const.rho * const.c_p)

        def compute_T_k(T_field):
            return self.compute_species_rhs(T_field, u, v, const.nu, T_source)

        k1 = compute_T_k(T)
        T2 = np.copy(T); T2[i_int,j_int] = T[i_int,j_int] + 0.5*dt*k1
        self.temperature_boundary(T2)
        k2 = compute_T_k(T2)

        T3 = np.copy(T); T3[i_int,j_int] = T[i_int,j_int] + 0.5*dt*k2
        self.temperature_boundary(T3)
        k3 = compute_T_k(T3)

        T4 = np.copy(T); T4[i_int,j_int] = T[i_int,j_int] + dt*k3
        self.temperature_boundary(T4)
        k4 = compute_T_k(T4)

        T[i_int,j_int] = T[i_int,j_int] + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # keep heated rod as before for initial transient
        if self.current_time <= self.Lx / self.Uslot:
            x = np.linspace(0.0, self.Lx, self.n)
            y = np.linspace(0.0, self.Ly, self.m)
            X, Y = np.meshgrid(x, y, indexing='ij')
            mask = np.abs(Y - self.Ly/2) <= self.rode / 3
            T[mask] = self.T_rode

        self.temperature_boundary(T)
        self.T = T

    def step(self):
        """
        Perform one time step of the simulation.
        
        Uses fractional step method (projection method):
        1. Predict velocities without pressure (u*, v*)
        2. Solve pressure Poisson equation
        3. Correct velocities to satisfy incompressibility
        4. Update scalars (T, species mass fractions)
        """
        # === STEP 1: PREDICT VELOCITIES (u*, v*) ===
        
        # Update interior velocity components using advection-diffusion
        u_star, v_star = self.fluid.adv_diff_interior(
            self.inv_dx, self.inv_dy, self.dt)
    
        # --- Left boundary (i=0) special handling ---
        i = 0
        for j in range(1, self.m-1):
            v_loc = self.fluid.v[i, j]
            u_star[i, j] = 0  # No-slip on left boundary
            
            # Upwind advection for v-component
            if v_loc >= 0:
                adv_v_y = v_loc * (v_loc - self.fluid.v[i, j-1]) / self.dy
            else:
                adv_v_y = v_loc * (self.fluid.v[i, j+1] - v_loc) / self.dy
            
            # Asymmetric diffusion (one-sided in x-direction)
            diffusion_v = self.fluid.diffusivity * (
                (2*self.fluid.v[i+1, j] - 2*v_loc) / self.dx**2 +
                (self.fluid.v[i, j+1] - 2*v_loc + self.fluid.v[i, j-1]) / self.dy**2
            )
            
            v_star[i, j] = v_loc + self.dt * (-adv_v_y + diffusion_v)

        self.fluid.apply_velocity_bcs(u_star, v_star, self.ind_inlet, self.ind_coflow, self.Uslot, self.Ucoflow)

        # === STEP 2: SOLVE PRESSURE POISSON EQUATION ===
        self.fluid.SOR_pressure_solver(u_star, v_star)
        
        # === STEP 3: CORRECT VELOCITIES WITH PRESSURE GRADIENT ===
        u_new, v_new = self.fluid.correction_velocity(u_star, v_star)
        self.fluid.apply_velocity_bcs(u_new, v_new, self.ind_inlet, self.ind_coflow, self.Uslot, self.Ucoflow)

        # === STEP 4: UPDATE SCALARS (TEMPERATURE, SPECIES) ===
        # Initialiser les taux de réaction
        self.ChemicalManager.update_reaction_rates(self.T)
        heat = self.ChemicalManager.heat_release()

        # Calcul du pas de temps chimique
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = (const.c_p * const.rho * self.T) / (np.abs(heat) + 1e-30)
        
        valid = tau[np.isfinite(tau) & (tau > 0)]
        if valid.size > 0:
            chemical_dt = 0.1 * np.min(valid)
            chemical_dt = np.clip(chemical_dt, 1e-10, self.dt)
        else:
            chemical_dt = self.dt

        # Boucle de sous-pas avec mise à jour des taux
        if chemical_dt < self.dt and chemical_dt > 0:
            chemical_time = 0.0
            n_substeps = 0
            max_substeps = 1000  # Sécurité contre boucles infinies
            
            while chemical_time < self.dt and n_substeps < max_substeps:
                # IMPORTANT: Recalculer les taux à chaque sous-pas
                self.ChemicalManager.update_reaction_rates(self.T)
                
                self.ChemicalManager.rk4_advance_species(u_new, v_new, chemical_dt, self.ind_inlet, self.ind_coflow)
                chemical_time += chemical_dt
                n_substeps += 1
                
                # Optionnel: ajuster le dernier pas pour arriver exactement à self.dt
                if chemical_time + chemical_dt > self.dt:
                    chemical_dt = self.dt - chemical_time
        else:
            self.ChemicalManager.rk4_advance_species(u_new, v_new, self.dt, self.ind_inlet, self.ind_coflow)
        
        # Mettre à jour la température avec les nouveaux taux
        self.ChemicalManager.update_reaction_rates(self.T)
        heat = self.ChemicalManager.heat_release()
        self.rk4_advance_temperature(u_new, v_new, heat, self.dt)
        
        self.fluid.u = u_new
        self.fluid.v = v_new
        
        # Advance time
        self.current_time += self.dt

    # ==================== Main run loop ====================

    def run(self, parallel=False):
        """
        Run the simulation until total_time is reached.
        
        Displays progress and saves results at the end.
        """
        start_time = time.time()
        time_data = 0.0
        count_steps = 0
        
        if parallel:
            function_step = self.step_numba
        else:
            function_step = self.step

        while self.current_time < self.total_time:
            function_step()
            progress = self.current_time / self.total_time * 100
            print(f"\rSimulation progress: {progress:.2f}%", end="")
            
            time_data += self.dt
            if time_data >= self.dt_data:
                time_data = 0.0
                # Format avec zéros pour tri correct: t0000, t0001, etc.
                self.save_dataset(filename_prefix=f"simulation_data_t{count_steps:04d}")
                count_steps += 1
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nSimulation completed!")
        print(f"Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        self.save_dataset()


    def print_caracteristics(self):
        """Print simulation parameters and grid characteristics."""
        print("=" * 50)
        print("SIMULATION CHARACTERISTICS")
        print("=" * 50)
        print(f"Time step:     {self.dt:.6e} s")
        print(f"Total time:    {self.total_time} s")
        print(f"Grid size:     {self.n} x {self.m}")
        print(f"Domain size:   {self.Lx*1e3:.2f} mm x {self.Ly*1e3:.2f} mm")
        print(f"Spatial steps: dx = {self.dx*1e3:.4f} mm, dy = {self.dy*1e3:.4f} mm")
        print("=" * 50)

        
    def save_dataset(self, filename_prefix="simulation_data"):
        """
        Save simulation results to .npy files.
        
        Args:
            filename_prefix: Prefix for output filenames
        """
        data_dir = "data//"
        
        # Save velocity fields
        np.save(data_dir + f"{filename_prefix}_u.npy", self.fluid.u)
        np.save(data_dir + f"{filename_prefix}_v.npy", self.fluid.v)
        
        # Save temperature field
        np.save(data_dir + f"{filename_prefix}_T.npy", self.T)
        
        # Save species mass fractions
        for name, chem in self.ChemicalManager.chemistries.items():
            np.save(data_dir + f"{filename_prefix}_Y_{name}.npy", chem.Y)
        
        # print(f"\nSimulation data saved with prefix '{filename_prefix}'.")


    def plot_concentration(self, load_from="data//simulation_data", save=False, filename="concentrations.png"):
        """
        Plot mass fraction distributions for all 5 species.
        
        Args:
            load_from: Path prefix to load data from
            save: Whether to save figure to file
            filename: Output filename if save=True
        """
        # === Load or use current species data ===
        def load_species_data(species_name):
            filepath = f"{load_from}_Y_{species_name}.npy"
            if os.path.exists(filepath):
                return np.load(filepath)
            else:
                return self.ChemicalManager.chemistries[species_name].Y
        
        Y_O2 = load_species_data('O2')
        Y_CH4 = load_species_data('CH4')
        Y_CO2 = load_species_data('CO2')
        Y_H2O = load_species_data('H2O')
        Y_N2 = load_species_data('N2')
        
        # === Setup figure with GridSpec (2 rows, 3 columns) ===
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Species data and colormaps
        species_data = [
            ('O₂', Y_O2, 'Blues'),
            ('CH₄', Y_CH4, 'Oranges'),
            ('CO₂', Y_CO2, 'Greens'),
            ('H₂O', Y_H2O, 'Purples'),
            ('N₂', Y_N2, 'Greys')
        ]
        
        # === Create subplots ===
        for idx, (name, Y, cmap) in enumerate(species_data):
            # Position plots in grid
            if idx < 3:
                ax = fig.add_subplot(gs[0, idx])
            else:
                ax = fig.add_subplot(gs[1, idx-3])
            
            # Plot mass fraction field
            im = ax.imshow(Y.T, extent=(0, self.Lx*1e3, 0, self.Ly*1e3), 
                           aspect='auto', cmap=cmap, origin='lower')
            
            ax.set_xlabel('x (mm)')
            ax.set_ylabel('y (mm)')
            ax.set_title(f'Mass Fraction of {name}')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label=f'Y_{name}')
        
        ax = fig.add_subplot(gs[1, 2])
        sum_Y = Y_O2 + Y_CH4 + Y_CO2 + Y_H2O + Y_N2
        im = ax.imshow(sum_Y.T, extent=(0, self.Lx*1e3, 0, self.Ly*1e3), 
                       aspect='auto', cmap='viridis', origin='lower')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title('Sum of Mass Fractions')
        plt.colorbar(im, ax=ax, label='Sum Y_i')

        # Overall title
        fig.suptitle('Species Mass Fraction Distributions', fontsize=16, fontweight='bold')
        
        if save:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        plt.show()
 
    def plot_temperature(self, load_from="data//simulation_data", save=False, filename="temperature.png"):
        """
        Plot temperature field distribution.
        
        Args:
            load_from: Path prefix to load data from
            save: Whether to save figure to file
            filename: Output filename if save=True
        """
        # Load temperature data
        try:
            T_plot = np.load(f"{load_from}_T.npy")
        except Exception:
            T_plot = self.T

        # Create figure
        fig = plt.figure(figsize=(8, 6))
        
        # Plot temperature field
        plt.imshow(T_plot.T, extent=(0, self.Lx*1e3, 0, self.Ly*1e3), 
                   aspect='auto', cmap='hot', origin = "lower")
        
        plt.colorbar(label='Temperature (K)')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Temperature Distribution')
        plt.tight_layout()
        
        if save:
            plt.savefig(filename)
        
        plt.show()

    def plot_velocity_magnitude(self, load_from="data//simulation_data", save=False, filename="velocity_magnitude.png"):
        """
        Plot velocity magnitude with vector field overlay.
        
        Args:
            load_from: Path prefix to load data from
            save: Whether to save figure to file
            filename: Output filename if save=True
        """
        # Load velocity data
        try:
            u_plot = np.load(f"{load_from}_u.npy")
        except Exception:
            u_plot = self.fluid.u
            
        try:
            v_plot = np.load(f"{load_from}_v.npy")
        except Exception:
            v_plot = self.fluid.v
        
        # Calculate velocity magnitude
        velocity_magnitude = np.sqrt(u_plot**2 + v_plot**2)
        
        # Create figure
        fig = plt.figure(figsize=(8, 6))
        
        # Plot velocity magnitude as contour
        plt.imshow(velocity_magnitude.T, extent=(0, self.Lx*1e3, 0, self.Ly*1e3), 
                   aspect='auto', cmap='viridis')
        
        plt.colorbar(label='Velocity Magnitude (m/s)')
        
        # Add velocity vector field overlay
        step = int(self.n / 20)
        x_coords = np.linspace(0, self.Lx*1e3, self.n)[::step]
        y_coords = np.linspace(0, self.Ly*1e3, self.m)[::step]
        
        plt.quiver(x_coords, y_coords,
                   u_plot[::step, ::step].T, v_plot[::step, ::step].T,
                   color='black', scale=50)
        
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Velocity Magnitude Distribution')
        plt.tight_layout()
        
        if save:
            plt.savefig(filename)
        
        plt.show()

    def animation_concentration(self, load_from="simulation_data", species='CH4', interval=200, save=False, filename="concentration_animation.gif"):
        """
        Create an animation of species concentration over time.
        
        Args:
            load_from: Path prefix to load data from
            species: Species name to animate (e.g., 'CH4')
            interval: Delay between frames in milliseconds
            save: Whether to save the animation to file
            filename: Output filename if save=True
        """
        print("Creating animation...")
        
        # Dictionnaire des colormaps par espèce (cohérent avec plot_concentration)
        species_colormaps = {
            'O2': 'Blues',
            'CH4': 'Oranges',
            'CO2': 'Greens',
            'H2O': 'Purples',
            'N2': 'Greys'
        }
        
        # Dictionnaire des noms d'affichage avec indices Unicode
        species_display_names = {
            'O2': 'O₂',
            'CH4': 'CH₄',
            'CO2': 'CO₂',
            'H2O': 'H₂O',
            'N2': 'N₂'
        }
        
        # Sélectionner la colormap appropriée (par défaut 'viridis')
        cmap = species_colormaps.get(species, 'viridis')
        display_name = species_display_names.get(species, species)
        
        # Tri naturel des fichiers
        file_list = sorted([f for f in os.listdir("data//") 
                       if f.startswith(f"{load_from}_t") and f.endswith(f'Y_{species}.npy')],
                       key=lambda x: int(''.join(filter(str.isdigit, x.split('_t')[1].split('_')[0]))))
        
        data_sequence = [np.load(os.path.join("data//", f)) for f in file_list]

        fig, ax = plt.subplots(figsize=(8, 6))
        # Utiliser la colormap spécifique à l'espèce
        im = ax.imshow(data_sequence[0].T, extent=(0, self.Lx*1e3, 0, self.Ly*1e3), 
                       aspect='auto', cmap=cmap, origin='lower', vmin=np.min(data_sequence[-1]), vmax=np.max(data_sequence[-1]), animated=True)
        title_text = ax.set_title(f'Mass Fraction of {display_name} at t=0s', animated=True)
        plt.colorbar(im, ax=ax, label=f'Y_{display_name}')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

        def update(frame):
            im.set_array(data_sequence[frame].T)
            # update animated title text
            title_text.set_text(f'Mass Fraction of {display_name} at t={frame*self.dt_data:.4f}s')
            return [im, title_text]

        ani = animation.FuncAnimation(fig, update, frames=len(data_sequence), interval=interval, blit=True)

        if save:
            ani.save(filename, writer='pillow', fps=30)

        print("Animation complete.")
        plt.show()
    
    def animation_temperature(self, load_from="simulation_data", interval=200, save=False, filename="temperature_animation.gif"):
        """
        Create an animation of temperature field over time.
        
        Args:
            load_from: Path prefix to load data from
            interval: Delay between frames in milliseconds
            save: Whether to save the animation to file
            filename: Output filename if save=True
        """
        print("Creating temperature animation...")
        
        # Tri naturel des fichiers
        file_list = sorted([f for f in os.listdir("data//") 
                       if f.startswith(f"{load_from}_t") and f.endswith('_T.npy')],
                       key=lambda x: int(''.join(filter(str.isdigit, x.split('_t')[1].split('_')[0]))))
        
        data_sequence = [np.load(os.path.join("data//", f)) for f in file_list]

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(data_sequence[0].T, extent=(0, self.Lx*1e3, 0, self.Ly*1e3), 
                       aspect='auto', cmap='hot', origin='lower', vmin=np.min(data_sequence[-1]), vmax=np.max(data_sequence[-1]), animated=True)
        title_text = ax.set_title(f'Temperature at t=0s', animated=True)
        plt.colorbar(im, ax=ax, label='Temperature (K)')
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

        def update(frame):
            im.set_array(data_sequence[frame].T)
            # update animated title text
            title_text.set_text(f'Temperature at t={frame*self.dt_data:.4f}s')
            return [im, title_text]

        ani = animation.FuncAnimation(fig, update, frames=len(data_sequence), interval=interval, blit=True)

        if save:
            ani.save(filename, writer='pillow', fps=30)

        print("Temperature animation complete.")
        plt.show()

    # ==================== Numba-accelerated versions ====================

    def temperature_boundary_numba(self, T_new):
        """Apply temperature boundary conditions using numba for parallel execution."""
        _apply_temperature_boundaries_numba(
            T_new, self.n, self.m, 
            self.ind_inlet, self.ind_coflow,
            self.Tslot, self.Tcoflow
        )

    def compute_species_rhs_numba(self, phi, u, v, diffusion_coef, source):
        """Compute RHS for species using numba for parallel execution."""
        rhs = np.zeros((self.n-4, self.m-4))
        _compute_species_rhs_numba(
            phi, u, v, diffusion_coef, 
            source if source is not None else np.zeros_like(phi),
            self.inv_dx, self.inv_dy, rhs
        )
        return rhs

    def rk4_advance_temperature_numba(self, u, v, heat, dt):
        """Advance temperature using RK4 with numba-accelerated computations."""
        T = np.copy(self.T)
        i_int = slice(2, -2); j_int = slice(2, -2)
        T_source = heat / (const.rho * const.c_p)

        def compute_T_k(T_field):
            return self.compute_species_rhs_numba(T_field, u, v, const.nu, T_source)

        k1 = compute_T_k(T)
        T2 = np.copy(T); T2[i_int,j_int] = T[i_int,j_int] + 0.5*dt*k1
        self.temperature_boundary_numba(T2)
        k2 = compute_T_k(T2)

        T3 = np.copy(T); T3[i_int,j_int] = T[i_int,j_int] + 0.5*dt*k2
        self.temperature_boundary_numba(T3)
        k3 = compute_T_k(T3)

        T4 = np.copy(T); T4[i_int,j_int] = T[i_int,j_int] + dt*k3
        self.temperature_boundary_numba(T4)
        k4 = compute_T_k(T4)

        T[i_int,j_int] = T[i_int,j_int] + dt/6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)

        # keep heated rod as before for initial transient
        if self.current_time <= self.Lx / self.Uslot:
            x = np.linspace(0.0, self.Lx, self.n)
            y = np.linspace(0.0, self.Ly, self.m)
            X, Y = np.meshgrid(x, y, indexing='ij')
            mask = np.abs(Y - self.Ly/2) <= self.rode / 3
            T[mask] = self.T_rode

        self.temperature_boundary_numba(T)
        self.T = T

    def step_numba(self):
        """
        Perform one time step of the simulation using numba-accelerated methods.
        
        Uses fractional step method (projection method):
        1. Predict velocities without pressure (u*, v*)
        2. Solve pressure Poisson equation
        3. Correct velocities to satisfy incompressibility
        4. Update scalars (T, species mass fractions)
        
        This version uses numba-compiled functions for improved performance.
        """
        # === STEP 1: PREDICT VELOCITIES (u*, v*) ===
        
        # Update interior velocity components using advection-diffusion (numba version)
        u_star, v_star = self.fluid.adv_diff_interior_numba(
            self.inv_dx, self.inv_dy)
    
        # --- Left boundary (i=0) special handling ---
        i = 0
        for j in range(1, self.m-1):
            v_loc = self.fluid.v[i, j]
            u_star[i, j] = 0  # No-slip on left boundary
            
            # Upwind advection for v-component
            if v_loc >= 0:
                adv_v_y = v_loc * (v_loc - self.fluid.v[i, j-1]) / self.dy
            else:
                adv_v_y = v_loc * (self.fluid.v[i, j+1] - v_loc) / self.dy
            
            # Asymmetric diffusion (one-sided in x-direction)
            diffusion_v = self.fluid.diffusivity * (
                (2*self.fluid.v[i+1, j] - 2*v_loc) / self.dx**2 +
                (self.fluid.v[i, j+1] - 2*v_loc + self.fluid.v[i, j-1]) / self.dy**2
            )
            
            v_star[i, j] = v_loc + self.dt * (-adv_v_y + diffusion_v)

        self.fluid.apply_velocity_bcs(u_star, v_star, self.ind_inlet, self.ind_coflow, self.Uslot, self.Ucoflow)

        # === STEP 2: SOLVE PRESSURE POISSON EQUATION (numba version) ===
        self.fluid.SOR_pressure_solver_numba(u_star, v_star)
        
        # === STEP 3: CORRECT VELOCITIES WITH PRESSURE GRADIENT (numba version) ===
        u_new, v_new = self.fluid.correction_velocity_numba(u_star, v_star)
        self.fluid.apply_velocity_bcs(u_new, v_new, self.ind_inlet, self.ind_coflow, self.Uslot, self.Ucoflow)

        # === STEP 4: UPDATE SCALARS (TEMPERATURE, SPECIES) ===
        # Initialiser les taux de réaction (numba version)
        self.ChemicalManager.update_reaction_rates_numba(self.T)
        heat = self.ChemicalManager.heat_release_numba()

        # Calcul du pas de temps chimique
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = (const.c_p * const.rho * self.T) / (np.abs(heat) + 1e-30)
        
        valid = tau[np.isfinite(tau) & (tau > 0)]
        if valid.size > 0:
            chemical_dt = 0.1 * np.min(valid)
            chemical_dt = np.clip(chemical_dt, 1e-10, self.dt)
        else:
            chemical_dt = self.dt

        # Boucle de sous-pas avec mise à jour des taux (numba versions)
        if chemical_dt < self.dt and chemical_dt > 0:
            chemical_time = 0.0
            n_substeps = 0
            max_substeps = 1000  # Sécurité contre boucles infinies
            
            while chemical_time < self.dt and n_substeps < max_substeps:
                # IMPORTANT: Recalculer les taux à chaque sous-pas
                self.ChemicalManager.update_reaction_rates_numba(self.T)
                
                self.ChemicalManager.rk4_advance_species_numba(u_new, v_new, chemical_dt, self.ind_inlet, self.ind_coflow)
                chemical_time += chemical_dt
                n_substeps += 1
                
                # Optionnel: ajuster le dernier pas pour arriver exactement à self.dt
                if chemical_time + chemical_dt > self.dt:
                    chemical_dt = self.dt - chemical_time
        else:
            self.ChemicalManager.rk4_advance_species_numba(u_new, v_new, self.dt, self.ind_inlet, self.ind_coflow)
        
        # Mettre à jour la température avec les nouveaux taux (numba versions)
        self.ChemicalManager.update_reaction_rates_numba(self.T)
        heat = self.ChemicalManager.heat_release_numba()
        self.rk4_advance_temperature_numba(u_new, v_new, heat, self.dt)
        
        self.fluid.u = u_new
        self.fluid.v = v_new
        
        # Advance time
        self.current_time += self.dt

# ==================== Numba-compiled functions ====================

@njit(parallel=True)
def _apply_temperature_boundaries_numba(T_new, n, m, ind_inlet, ind_coflow, Tslot, Tcoflow):
    """Numba-compiled function to apply temperature boundary conditions in parallel."""
    # Left boundary (i=0,1) - Neumann
    for j in prange(2, m-2):
        T_new[1, j] = T_new[2, j]
        T_new[0, j] = T_new[1, j]
    
    # CH4 inlet (slot region, bottom wall j=0,1)
    for i in prange(ind_inlet):
        T_new[i, 0] = Tslot
        T_new[i, 1] = Tslot
    
    # O2+N2 inlet (slot region, top wall)
    for i in prange(ind_inlet):
        T_new[i, m-1] = Tslot
        T_new[i, m-2] = Tslot
    
    # N2 coflow inlet (coflow region, bottom wall)
    for i in prange(ind_inlet, ind_coflow):
        T_new[i, 0] = Tcoflow
        T_new[i, 1] = Tcoflow
    
    # N2 coflow inlet (coflow region, top wall)
    for i in prange(ind_inlet, ind_coflow):
        T_new[i, m-1] = Tcoflow
        T_new[i, m-2] = Tcoflow
    
    # Lower wall (outlet region) - Neumann
    for i in prange(ind_coflow, n):
        T_new[i, 1] = T_new[i, 2]
        T_new[i, 0] = T_new[i, 2]
    
    # Upper wall (outlet region) - Neumann
    for i in prange(ind_coflow, n):
        T_new[i, m-2] = T_new[i, m-3]
        T_new[i, m-1] = T_new[i, m-3]
    
    # Right boundary (outlet) - Extrapolation
    for j in prange(2, m-2):
        T_new[n-2, j] = T_new[n-3, j]
        T_new[n-1, j] = T_new[n-3, j]


@njit(parallel=True)
def _compute_species_rhs_numba(phi, u, v, diffusion_coef, source, inv_dx, inv_dy, rhs):
    """Numba-compiled function to compute species RHS in parallel."""
    n_interior = rhs.shape[0]
    m_interior = rhs.shape[1]
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    
    for i in prange(n_interior):
        for j in range(m_interior):
            ii = i + 2  # offset to actual grid position
            jj = j + 2
            
            phi_loc = phi[ii, jj]
            phi_m2_x = phi[ii-2, jj]
            phi_m1_x = phi[ii-1, jj]
            phi_p1_x = phi[ii+1, jj]
            phi_p2_x = phi[ii+2, jj]
            phi_m2_y = phi[ii, jj-2]
            phi_m1_y = phi[ii, jj-1]
            phi_p1_y = phi[ii, jj+1]
            phi_p2_y = phi[ii, jj+2]
            
            u_loc = u[ii, jj]
            v_loc = v[ii, jj]
            
            # Upwind advection
            if u_loc > 0:
                adv_x = u_loc * (phi_loc - phi_m1_x) * inv_dx
            else:
                adv_x = u_loc * (phi_p1_x - phi_loc) * inv_dx
            
            if v_loc > 0:
                adv_y = v_loc * (phi_loc - phi_m1_y) * inv_dy
            else:
                adv_y = v_loc * (phi_p1_y - phi_loc) * inv_dy
            
            # 4th-order central diffusion
            diff_x = (-phi_p2_x + 16.0*phi_p1_x - 30.0*phi_loc + 16.0*phi_m1_x - phi_m2_x) * inv_dx2 / 12.0
            diff_y = (-phi_p2_y + 16.0*phi_p1_y - 30.0*phi_loc + 16.0*phi_m1_y - phi_m2_y) * inv_dy2 / 12.0
            
            diff = diffusion_coef * (diff_x + diff_y)
            src = source[ii, jj]
            
            rhs[i, j] = -adv_x - adv_y + diff + src