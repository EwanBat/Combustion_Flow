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

        # === Physical constants from constant.py ===
        self.visc = const.alpha
        self.rho = const.rho
        self.c_p = const.c_p

        # === Domain physical dimensions (m) ===
        self.Lx = const.Lx              # Domain length in x
        self.Ly = const.Ly              # Domain width in y
        self.Lslot = const.Lslot         # Length of fuel/oxidizer slot
        self.Lcoflow = const.Lcoflow       # Length of coflow region
    
        # === Spatial discretization ===
        self.dx = self.Lx / (self.n - 1)
        self.dy = self.Ly / (self.m - 1)
        self.inv_dx = 1.0 / self.dx
        self.inv_dy = 1.0 / self.dy

        # === Inlet boundary conditions ===
        self.Uslot = const.Uslot            # Slot inlet velocity (m/s)
        self.Tslot = const.Tslot          # Slot inlet temperature (K)
        self.Ucoflow = const.Ucoflow          # Coflow inlet velocity (m/s)
        self.Tcoflow = const.Tcoflow        # Coflow inlet temperature (K)

        # === Index markers for different boundary regions ===
        # Ajout d'un léger décalage pour éviter les résonances numériques quand ind_coflow = n/2
        self.ind_inlet = int(self.Lslot / self.dx)                     # End of inlet slot
        self.ind_coflow = int((self.Lslot + self.Lcoflow) / self.dx)   # End of coflow region
        
        # Si ind_coflow tombe exactement sur n/2, décaler légèrement pour éviter les résonances
        if abs(self.ind_coflow - self.n/2) < 1.5:
            self.ind_coflow = int(self.n/2) + 2  # Décalage de 2 cellules
        
        # === Time step calculation based on stability criteria ===
        # CFL condition for advection: Δt ≤ CFL * Δx / U
        # Fourier condition for diffusion: Δt ≤ Fo * Δx² / ν
        # Réduction du CFL pour grilles critiques (éviter résonances numériques)
        self.CFL = 0.2 if (n % 50 == 2 or abs(n/2 - round(n/2)) < 0.1) else 0.25
        self.Fo = 0.12 if (n % 50 == 2 or abs(n/2 - round(n/2)) < 0.1) else 0.15
        dt_cfl = self.CFL * np.min((self.dx, self.dy)) / self.Uslot
        dt_fourier = self.Fo * np.min((self.dx, self.dy))**2 / np.max((const.nu, const.alpha, const.D))
        self.dt = np.min((dt_cfl, dt_fourier))

        # === Temperature field initialization === Circular heated rod in center
        self.rode = const.width_rode # Width of the heating rode (m)
        self.T_rode = const.T_rode  # Temperature of the heated rod (K)
        self.T = np.ones((self.n, self.m)) * 300        # Temperature field (K) with heated rod of size self.rode centered in domain
        x = np.linspace(0.0, self.Lx, self.n)
        y = np.linspace(0.0, self.Ly, self.m)
        X, Y = np.meshgrid(x, y, indexing='ij')
        mask = np.abs(Y - self.Ly/2) <= self.rode
        self.T[mask] = self.T_rode

        # === Physical fields ===
        self.fluid = fluid
        self.ChemicalManager = ChemicalManager

        self.fluid.initiate_steps(self.dt, self.dx, self.dy)
        self.ChemicalManager.initiate_steps(self.dt, self.dx, self.dy)

        self.fluid.BC_initialization(self.ind_inlet, self.ind_coflow, self.Uslot, self.Ucoflow)

    def update_dt(self):
        """Recompute the time step based on current velocity field and stability criteria."""
        max_u = np.max(np.abs(self.fluid.u))
        max_v = np.max(np.abs(self.fluid.v))
        max_velocity = max(max_u, max_v, 1e-10)  # Avoid division by zero

        dt_cfl = self.CFL * np.min((self.dx, self.dy)) / max_velocity
        dt_fourier = self.Fo * np.min((self.dx, self.dy))**2 / np.max((const.nu, const.alpha, const.D))
        self.dt = np.min((dt_cfl, dt_fourier))

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
        """
        Advance temperature field using RK4 integrator.
        Args:
            u, v (np.ndarray): velocity fields in x and y directions.
            heat (np.ndarray): volumetric heat release field (W/m^3).
            dt (float): time-step size.
        """
        # prepare array
        T = np.copy(self.T)
        i_int = slice(2, -2); j_int = slice(2, -2)
        T_source = heat / (self.rho * self.c_p)

        def compute_T_k(T_field):
            return self.compute_species_rhs(T_field, u, v, self.visc, T_source)

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
            mask = np.abs(Y - self.Ly/2) <= self.rode
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
        u_star, v_star = self.fluid.adv_diff_interior()
        self.fluid.apply_velocity_bcs(self.fluid.u, self.fluid.v, u_star, v_star)

        # === STEP 2: SOLVE PRESSURE POISSON EQUATION ===
        self.fluid.SOR_pressure_solver(u_star, v_star)
        
        # === STEP 3: CORRECT VELOCITIES WITH PRESSURE GRADIENT ===
        u_new, v_new = self.fluid.correction_velocity(u_star, v_star)
        self.fluid.apply_velocity_bcs(u_star, v_star, u_new, v_new, True)

        # === STEP 4: UPDATE SCALARS (TEMPERATURE, SPECIES) ===
        # Initialiser les taux de réaction
        self.ChemicalManager.update_reaction_rates(self.T)
        heat = self.ChemicalManager.heat_release()

        # Calcul du pas de temps chimique
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = (self.c_p * self.rho * self.T) / (np.abs(heat) + 1e-30)
        
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
        T_source = heat / (self.rho * self.c_p)

        def compute_T_k(T_field):
            return self.compute_species_rhs_numba(T_field, u, v, self.visc, T_source)

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
        if self.current_time <= self.Lx / self.Ucoflow:
            x = np.linspace(0.0, self.Lx, self.n)
            y = np.linspace(0.0, self.Ly, self.m)
            X, Y = np.meshgrid(x, y, indexing='ij')
            mask = np.abs(Y - self.Ly/2) <= self.rode
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
        self.fluid.apply_velocity_bcs(self.fluid.u, self.fluid.v, u_star, v_star)

        # === STEP 2: SOLVE PRESSURE POISSON EQUATION (numba version) ===
        self.fluid.SOR_pressure_solver_numba(u_star, v_star)
        
        # === STEP 3: CORRECT VELOCITIES WITH PRESSURE GRADIENT (numba version) ===
        u_new, v_new = self.fluid.correction_velocity_numba(u_star, v_star)
        self.fluid.apply_velocity_bcs(u_star, v_star, u_new, v_new, True)

        # === STEP 4: UPDATE SCALARS (TEMPERATURE, SPECIES) ===
        # Initialiser les taux de réaction (numba version)
        self.ChemicalManager.update_reaction_rates_numba(self.T)
        heat = self.ChemicalManager.heat_release_numba()

        # Calcul du pas de temps chimique
        with np.errstate(divide='ignore', invalid='ignore'):
            tau = (self.c_p * self.rho * self.T) / (np.abs(heat) + 1e-30)
        
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