import numpy as np
import constant as const
from numba import njit, prange

class Chemistry:
    """
    Lightweight container representing a single chemical species.

    This class stores basic physical properties for one species (density,
    molar mass and diffusivity) and holds the mass-fraction field Y.
    It is intentionally minimal: numerical operations and boundary handling
    are performed by the ChemistryManager.

    Attributes:
        density (float): species density (kg/m^3).
        molar_mass (float): species molar mass (kg/mol).
        diffusivity (float): species molecular diffusivity (m^2/s).
        Y (np.ndarray): mass fraction field (initialized via Y_initialization).
    """
    def __init__(self, density, molar_mass, diffusivity):
        """
        Initialize a Chemistry object.

        Args:
            density (float): species density (kg/m^3).
            molar_mass (float): species molar mass (kg/mol).
            diffusivity (float): molecular diffusivity (m^2/s).

        The constructor stores the provided physical parameters. The spatial
        mass-fraction field Y is not allocated here; use Y_initialization
        to attach a grid-sized array when the computational mesh is known.
        """
        self.density = density
        self.molar_mass = molar_mass
        self.diffusivity = diffusivity

    def Y_initialization(self, Y_initial):
        """Initialize the mass fraction field."""
        self.Y = np.array(Y_initial)

# Ajout : gestionnaire multi-espèces (chemistry manager)
class ChemistryManager:
    """
    Manager handling multiple chemical species and simple reaction kinetics.

    Responsibilities:
    - hold a dictionary of Chemistry objects (one per species)
    - store and update reaction rates on the computational grid
    - provide routines to initialize grid-dependent steps (inv_dx, inv_dy)
    - compute Arrhenius-based reaction rates and volumetric heat release
    - enforce species boundary conditions, compute RHS for advection-diffusion
      and advance species using an RK4 integrator with operator-splitting style.

    Important methods:
        initiate_steps(dx, dy): provide grid spacing inverses used by RHS.
        update_reaction_rates(T_field): compute temperature-dependent rates.
        heat_release(): compute Omega_T (volumetric heat source).
        rk4_advance_species(...): advance all species mass fractions one time-step.
    """

    def __init__(self, n, m, chem_dict: dict[str, Chemistry]):
        """
        Initialize the ChemistryManager.

        Args:
            n (int): number of grid points in x-direction.
            m (int): number of grid points in y-direction.
            chem_dict (dict): mapping species name -> Chemistry instance.

        The manager prepares per-species reaction rate arrays sized to match the
        species fields and stores grid dimensions. Grid-dependent inverses are
        set later via initiate_steps when dx and dy are known.
        """
        self.chemistries = chem_dict
        self.reaction_rates = {name: np.zeros_like(next(iter(chem_dict.values())).Y) for name in chem_dict}
        self.n = n
        self.m = m
        self.i_slice = slice(2, -2)
        self.j_slice = slice(2, -2)

    def initiate_steps(self, dx, dy):
        self.inv_dx = 1/dx
        self.inv_dy = 1/dy

    def update_reaction_rates(self, T_field):
        """Compute Arrhenius-based reaction rates for the global reaction.
        Args:
            T_field (np.ndarray): temperature field (K).
        Returns:
            reaction_rates (dict): mapping species name -> reaction rate array (kg/m^3/s).
        """
        # ============================================================================
        # SPECIES DATA: Extract chemistry objects for reactants
        # ============================================================================
        O2 = self.chemistries['O2']
        CH4 = self.chemistries['CH4']
        
        # ============================================================================
        # ARRHENIUS RATE: Calculate temperature-dependent rate constant k
        # ============================================================================
        exp_arg = -const.T_A / T_field
        k = const.A * np.exp(exp_arg)
        
        # ============================================================================
        # CONCENTRATIONS: Convert mass fractions to molar concentrations
        # ============================================================================
        O2_conc = O2.Y * const.rho / O2.molar_mass
        CH4_conc = CH4.Y * const.rho / CH4.molar_mass
        
        # ============================================================================
        # REACTION RATE: Compute volumetric reaction rate Q
        # ============================================================================
        Q = k * O2_conc**2 * CH4_conc
        
        # ============================================================================
        # SPECIES RATES: Calculate production/consumption rate for each species
        # ============================================================================
        reaction_rates = {}
        for name, chem in self.chemistries.items():
            nu_i = const.stoichiometric_coefficients.get(name, 0)
            rate = nu_i * Q * const.molar_mass[name]
            reaction_rates[name] = rate
        self.reaction_rates = reaction_rates
        return reaction_rates

    def heat_release(self):
        """Compute volumetric heat release Omega_T (W/m^3) from reaction rates.
        Returns:
            omega_T (np.ndarray): volumetric heat release field (W/m^3).
        """
        # ============================================================================
        # HEAT RELEASE: Compute volumetric heat source from reaction enthalpies
        # ============================================================================
        omega_T = 0.0
        for name, rate in self.reaction_rates.items():
            delta_h = const.enthalpy_of_formation.get(name, 0)
            omega_T -= rate * delta_h / self.chemistries[name].molar_mass
        return omega_T

    # Boundary conditions for species (mêmes BCs que dans System)
    def chemistry_boundaries(self, Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new, Y_N2_new, ind_inlet, ind_coflow):
        """
        Apply boundary conditions to species mass-fraction fields.
        Args:
            Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new, Y_N2_new (np.ndarray):mass-fraction fields to apply BCs on.
            ind_inlet (int): index separating inlet slot from coflow region.
            ind_coflow (int): index separating coflow region from outlet.
        """
        # ============================================================================
        # LEFT BOUNDARY: Neumann condition (zero gradient) for 4th-order stencil
        # ============================================================================
        for j in range(2, self.m-2):
            Y_CH4_new[0, j] = Y_CH4_new[2, j]
            Y_O2_new[0, j] = Y_O2_new[2, j]
            Y_CO2_new[0, j] = Y_CO2_new[2, j]
            Y_H2O_new[0, j] = Y_H2O_new[2, j]
            Y_N2_new[0, j] = Y_N2_new[2, j]
            Y_CH4_new[1, j] = Y_CH4_new[2, j]
            Y_O2_new[1, j] = Y_O2_new[2, j]
            Y_CO2_new[1, j] = Y_CO2_new[2, j]
            Y_H2O_new[1, j] = Y_H2O_new[2, j]
            Y_N2_new[1, j] = Y_N2_new[2, j]
        
        # ============================================================================
        # INLET CH4: Slot region bottom wall (pure CH4)
        # ============================================================================
        Y_CH4_new[:ind_inlet, 0] = 1.0
        Y_O2_new[:ind_inlet, 0] = 0.0
        Y_CO2_new[:ind_inlet, 0] = 0.0
        Y_H2O_new[:ind_inlet, 0] = 0.0
        Y_N2_new[:ind_inlet, 0] = 0.0
        Y_CH4_new[:ind_inlet, 1] = 1.0
        Y_O2_new[:ind_inlet, 1] = 0.0
        Y_CO2_new[:ind_inlet, 1] = 0.0
        Y_H2O_new[:ind_inlet, 1] = 0.0
        Y_N2_new[:ind_inlet, 1] = 0.0
        
        # ============================================================================
        # INLET O2+N2: Slot region top wall (air composition)
        # ============================================================================
        Y_CH4_new[:ind_inlet, self.m-1] = 0.0
        Y_O2_new[:ind_inlet, self.m-1] = 0.21
        Y_CO2_new[:ind_inlet, self.m-1] = 0.0
        Y_H2O_new[:ind_inlet, self.m-1] = 0.0
        Y_N2_new[:ind_inlet, self.m-1] = 0.79
        Y_CH4_new[:ind_inlet, self.m-2] = 0.0
        Y_O2_new[:ind_inlet, self.m-2] = 0.21
        Y_CO2_new[:ind_inlet, self.m-2] = 0.0
        Y_H2O_new[:ind_inlet, self.m-2] = 0.0
        Y_N2_new[:ind_inlet, self.m-2] = 0.79
        
        # ============================================================================
        # COFLOW N2: Coflow region bottom wall (pure N2)
        # ============================================================================
        Y_CH4_new[ind_inlet:ind_coflow, 0] = 0.0
        Y_O2_new[ind_inlet:ind_coflow, 0] = 0.0
        Y_CO2_new[ind_inlet:ind_coflow, 0] = 0.0
        Y_H2O_new[ind_inlet:ind_coflow, 0] = 0.0
        Y_N2_new[ind_inlet:ind_coflow, 0] = 1.0
        Y_CH4_new[ind_inlet:ind_coflow, 1] = 0.0
        Y_O2_new[ind_inlet:ind_coflow, 1] = 0.0
        Y_CO2_new[ind_inlet:ind_coflow, 1] = 0.0
        Y_H2O_new[ind_inlet:ind_coflow, 1] = 0.0
        Y_N2_new[ind_inlet:ind_coflow, 1] = 1.0
        
        # ============================================================================
        # COFLOW N2: Coflow region top wall (pure N2)
        # ============================================================================
        Y_CH4_new[ind_inlet:ind_coflow, self.m-1] = 0.0
        Y_O2_new[ind_inlet:ind_coflow, self.m-1] = 0.0
        Y_CO2_new[ind_inlet:ind_coflow, self.m-1] = 0.0
        Y_H2O_new[ind_inlet:ind_coflow, self.m-1] = 0.0
        Y_N2_new[ind_inlet:ind_coflow, self.m-1] = 1.0
        Y_CH4_new[ind_inlet:ind_coflow, self.m-2] = 0.0
        Y_O2_new[ind_inlet:ind_coflow, self.m-2] = 0.0
        Y_CO2_new[ind_inlet:ind_coflow, self.m-2] = 0.0
        Y_H2O_new[ind_inlet:ind_coflow, self.m-2] = 0.0
        Y_N2_new[ind_inlet:ind_coflow, self.m-2] = 1.0
        
        # ============================================================================
        # WALL BOUNDARY: Lower wall outlet region (Neumann condition)
        # ============================================================================
        Y_CH4_new[ind_coflow:, 0] = Y_CH4_new[ind_coflow:, 2]
        Y_O2_new[ind_coflow:, 0] = Y_O2_new[ind_coflow:, 2]
        Y_CO2_new[ind_coflow:, 0] = Y_CO2_new[ind_coflow:, 2]
        Y_H2O_new[ind_coflow:, 0] = Y_H2O_new[ind_coflow:, 2]
        Y_N2_new[ind_coflow:, 0] = Y_N2_new[ind_coflow:, 2]
        Y_CH4_new[ind_coflow:, 1] = Y_CH4_new[ind_coflow:, 2]
        Y_O2_new[ind_coflow:, 1] = Y_O2_new[ind_coflow:, 2]
        Y_CO2_new[ind_coflow:, 1] = Y_CO2_new[ind_coflow:, 2]
        Y_H2O_new[ind_coflow:, 1] = Y_H2O_new[ind_coflow:, 2]
        Y_N2_new[ind_coflow:, 1] = Y_N2_new[ind_coflow:, 2]
        
        # ============================================================================
        # WALL BOUNDARY: Upper wall outlet region (Neumann condition)
        # ============================================================================
        Y_CH4_new[ind_coflow:, self.m-1] = Y_CH4_new[ind_coflow:, self.m-3]
        Y_O2_new[ind_coflow:, self.m-1] = Y_O2_new[ind_coflow:, self.m-3]
        Y_CO2_new[ind_coflow:, self.m-1] = Y_CO2_new[ind_coflow:, self.m-3]
        Y_H2O_new[ind_coflow:, self.m-1] = Y_H2O_new[ind_coflow:, self.m-3]
        Y_N2_new[ind_coflow:, self.m-1] = Y_N2_new[ind_coflow:, self.m-3]
        Y_CH4_new[ind_coflow:, self.m-2] = Y_CH4_new[ind_coflow:, self.m-3]
        Y_O2_new[ind_coflow:, self.m-2] = Y_O2_new[ind_coflow:, self.m-3]
        Y_CO2_new[ind_coflow:, self.m-2] = Y_CO2_new[ind_coflow:, self.m-3]
        Y_H2O_new[ind_coflow:, self.m-2] = Y_H2O_new[ind_coflow:, self.m-3]
        Y_N2_new[ind_coflow:, self.m-2] = Y_N2_new[ind_coflow:, self.m-3]
        
        # ============================================================================
        # OUTLET BOUNDARY: Right boundary (extrapolation)
        # ============================================================================
        Y_CH4_new[self.n-1, 2:self.m-2] = Y_CH4_new[self.n-3, 2:self.m-2]
        Y_O2_new[self.n-1, 2:self.m-2] = Y_O2_new[self.n-3, 2:self.m-2]
        Y_CO2_new[self.n-1, 2:self.m-2] = Y_CO2_new[self.n-3, 2:self.m-2]
        Y_H2O_new[self.n-1, 2:self.m-2] = Y_H2O_new[self.n-3, 2:self.m-2]
        Y_N2_new[self.n-1, 2:self.m-2] = Y_N2_new[self.n-3, 2:self.m-2]
        Y_CH4_new[self.n-2, 2:self.m-2] = Y_CH4_new[self.n-3, 2:self.m-2]
        Y_O2_new[self.n-2, 2:self.m-2] = Y_O2_new[self.n-3, 2:self.m-2]
        Y_CO2_new[self.n-2, 2:self.m-2] = Y_CO2_new[self.n-3, 2:self.m-2]
        Y_H2O_new[self.n-2, 2:self.m-2] = Y_H2O_new[self.n-3, 2:self.m-2]
        Y_N2_new[self.n-2, 2:self.m-2] = Y_N2_new[self.n-3, 2:self.m-2]

    # Compute RHS for one species (vectorisé)
    def compute_species_rhs(self, phi, u, v, diffusion_coef, source):
        """
        Compute the RHS of the advection-diffusion-reaction equation for one species.
        Args:
            phi (np.ndarray): mass fraction field of the species.
            u, v (np.ndarray): velocity fields in x and y directions.
            diffusion_coef (float): molecular diffusivity of the species.
            source (np.ndarray or None): source term field (kg/kg/s) or None.
        """
        # ============================================================================
        # GRID SETUP: Define interior domain for 4th-order stencil
        # ============================================================================
        phi_loc = phi[self.i_slice, self.j_slice]
        
        # ============================================================================
        # FIELD EXTRACTION: Get shifted values for spatial derivatives
        # ============================================================================
        # Shifts in x-direction
        phi_m2_x = phi[:-4, self.j_slice]
        phi_m1_x = phi[1:-3, self.j_slice]
        phi_p1_x = phi[3:-1, self.j_slice]
        phi_p2_x = phi[4:, self.j_slice]
        # Shifts in y-direction
        phi_m2_y = phi[self.i_slice, :-4]
        phi_m1_y = phi[self.i_slice, 1:-3]
        phi_p1_y = phi[self.i_slice, 3:-1]
        phi_p2_y = phi[self.i_slice, 4:]

        # ============================================================================
        # ADVECTION: Compute upwind advective fluxes
        # ============================================================================
        u_loc = u[self.i_slice, self.j_slice]; v_loc = v[self.i_slice, self.j_slice]
        u_pos = np.maximum(u_loc, 0); u_neg = np.minimum(u_loc, 0)
        adv_x = (u_pos * (phi_loc - phi_m1_x) * self.inv_dx +
                 u_neg * (phi_p1_x - phi_loc) * self.inv_dx)
        v_pos = np.maximum(v_loc, 0); v_neg = np.minimum(v_loc, 0)
        adv_y = (v_pos * (phi_loc - phi_m1_y) * self.inv_dy +
                 v_neg * (phi_p1_y - phi_loc) * self.inv_dy)

        # ============================================================================
        # DIFFUSION: Compute 4th-order central differences (5-point stencil)
        # ============================================================================
        diff_x = (-phi_p2_x + 16.0*phi_p1_x - 30.0*phi_loc + 16.0*phi_m1_x - phi_m2_x) * self.inv_dx**2 / 12.0
        diff_y = (-phi_p2_y + 16.0*phi_p1_y - 30.0*phi_loc + 16.0*phi_m1_y - phi_m2_y) * self.inv_dy**2 / 12.0

        diff = diffusion_coef * (diff_x + diff_y)
        
        # ============================================================================
        # SOURCE TERM: Add chemical reaction source
        # ============================================================================
        src = source[self.i_slice, self.j_slice] if source is not None else 0.0
        
        return -adv_x - adv_y + diff + src

    def rk4_advance_species(self, u, v, dt, ind_inlet, ind_coflow):
        """Advance species using RK4 integrator.
        Args:
            u, v (np.ndarray): velocity fields in x and y directions.
            dt (float): time-step size.
            ind_inlet (int): index separating inlet slot from coflow region.
            ind_coflow (int): index separating coflow region from outlet.
        """
        # ============================================================================
        # INITIALIZATION: Prepare arrays and sources
        # ============================================================================
        Y = {name: np.copy(chem.Y) for name, chem in self.chemistries.items()}
        # sources in mass-fraction units (kg/m3 -> divide by rho inside RHS call as source expects kg/kg/s)
        sources = {name: (self.reaction_rates[name] / const.rho) for name in self.chemistries.keys()}

        # helper to compute all k given current full-field arrays
        def compute_all_k(Y_fields):
            ks = {}
            for name, chem in self.chemistries.items():
                ks[name] = self.compute_species_rhs(Y_fields[name], u, v, chem.diffusivity, sources[name])
            return ks

        # ============================================================================
        # RK4 STAGE 1: Compute k1 at current time
        # ============================================================================
        k1 = compute_all_k(Y)

        # ============================================================================
        # RK4 STAGE 2: Compute k2 at midpoint using k1
        # ============================================================================
        Y_k2 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k2[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + 0.5 * dt * k1[name]
        # apply BCs on temporary fields
        self.chemistry_boundaries(Y_k2['CH4'], Y_k2['O2'], Y_k2['CO2'], Y_k2['H2O'], Y_k2['N2'], ind_inlet, ind_coflow)
        k2 = compute_all_k(Y_k2)

        # ============================================================================
        # RK4 STAGE 3: Compute k3 at midpoint using k2
        # ============================================================================
        Y_k3 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k3[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + 0.5 * dt * k2[name]
        self.chemistry_boundaries(Y_k3['CH4'], Y_k3['O2'], Y_k3['CO2'], Y_k3['H2O'], Y_k3['N2'], ind_inlet, ind_coflow)
        k3 = compute_all_k(Y_k3)

        # ============================================================================
        # RK4 STAGE 4: Compute k4 at endpoint using k3
        # ============================================================================
        Y_k4 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k4[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + dt * k3[name]
        self.chemistry_boundaries(Y_k4['CH4'], Y_k4['O2'], Y_k4['CO2'], Y_k4['H2O'], Y_k4['N2'], ind_inlet, ind_coflow)
        k4 = compute_all_k(Y_k4)

        # ============================================================================
        # RK4 COMBINATION: Weighted average of all stages
        # ============================================================================
        for name in Y:
            Y[name][2:-2, 2:-2] = (Y[name][2:-2, 2:-2] +
                                   dt/6.0 * (k1[name] + 2.0*k2[name] + 2.0*k3[name] + k4[name]))

        # ============================================================================
        # FINALIZATION: Apply boundary conditions and clip small values
        # ============================================================================
        self.chemistry_boundaries(Y['CH4'], Y['O2'], Y['CO2'], Y['H2O'], Y['N2'], ind_inlet, ind_coflow)
        for name in Y:
            Y[name][Y[name] < 1e-15] = 0.0
            self.chemistries[name].Y = Y[name]

    # ==================== Numba-accelerated versions ====================

    def chemistry_boundaries_numba(self, Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new, Y_N2_new, ind_inlet, ind_coflow):
        """Apply boundary conditions using numba for parallel execution."""
        _apply_chemistry_boundaries_numba(
            Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new, Y_N2_new,
            ind_inlet, ind_coflow, self.n, self.m
        )

    def compute_species_rhs_numba(self, phi, u, v, diffusion_coef, source):
        """Compute RHS for one species using numba for parallel execution."""
        rhs = np.zeros((self.n-4, self.m-4))
        _compute_species_rhs_numba(
            phi, u, v, diffusion_coef, source if source is not None else np.zeros_like(phi),
            self.inv_dx, self.inv_dy, rhs
        )
        return rhs

    def update_reaction_rates_numba(self, T_field):
        """Update reaction rates using numba-accelerated computation."""
        # Convertir les dictionnaires en arrays pour numba
        stoich_array = np.array([
            const.stoichiometric_coefficients['CH4'],
            const.stoichiometric_coefficients['O2'],
            const.stoichiometric_coefficients['CO2'],
            const.stoichiometric_coefficients['H2O']
        ])
        
        molar_mass_array = np.array([
            const.molar_mass['CH4'],
            const.molar_mass['O2'],
            const.molar_mass['CO2'],
            const.molar_mass['H2O']
        ])
        
        k_CH4, k_O2, k_CO2, k_H2O = _compute_reaction_rates_numba(
            T_field,
            self.chemistries['O2'].Y,
            self.chemistries['CH4'].Y,
            self.chemistries['O2'].molar_mass,
            self.chemistries['CH4'].molar_mass,
            const.T_A,
            const.A,
            const.rho,
            stoich_array,
            molar_mass_array
        )
        
        self.reaction_rates = {
            'CH4': k_CH4,
            'O2': k_O2,
            'CO2': k_CO2,
            'H2O': k_H2O,
            'N2': np.zeros_like(k_CH4)  # N2 est inerte
        }
        
    def heat_release_numba(self):
        """Compute volumetric heat release using numba for parallel execution."""
        rates_array = np.array([self.reaction_rates[name] for name in self.chemistries.keys()])
        delta_h_array = np.array([const.enthalpy_of_formation.get(name, 0) for name in self.chemistries.keys()])
        molar_mass_array = np.array([self.chemistries[name].molar_mass for name in self.chemistries.keys()])
        
        omega_T = _compute_heat_release_numba(rates_array, delta_h_array, molar_mass_array)
        return omega_T


    def rk4_advance_species_numba(self, u, v, dt, ind_inlet, ind_coflow):
        """Advance species using RK4 with numba-accelerated computations."""
        # prepare arrays
        Y = {name: np.copy(chem.Y) for name, chem in self.chemistries.items()}
        sources = {name: (self.reaction_rates[name] / const.rho) for name in self.chemistries.keys()}
        diffusivities = {name: chem.diffusivity for name, chem in self.chemistries.items()}

        # helper to compute all k given current full-field arrays
        def compute_all_k(Y_fields):
            ks = {}
            for name, chem in self.chemistries.items():
                ks[name] = self.compute_species_rhs_numba(Y_fields[name], u, v, diffusivities[name], sources[name])
            return ks

        # k1
        k1 = compute_all_k(Y)

        # k2
        Y_k2 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k2[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + 0.5 * dt * k1[name]
        self.chemistry_boundaries_numba(Y_k2['CH4'], Y_k2['O2'], Y_k2['CO2'], Y_k2['H2O'], Y_k2['N2'], ind_inlet, ind_coflow)
        k2 = compute_all_k(Y_k2)

        # k3
        Y_k3 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k3[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + 0.5 * dt * k2[name]
        self.chemistry_boundaries_numba(Y_k3['CH4'], Y_k3['O2'], Y_k3['CO2'], Y_k3['H2O'], Y_k3['N2'], ind_inlet, ind_coflow)
        k3 = compute_all_k(Y_k3)

        # k4
        Y_k4 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k4[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + dt * k3[name]
        self.chemistry_boundaries_numba(Y_k4['CH4'], Y_k4['O2'], Y_k4['CO2'], Y_k4['H2O'], Y_k4['N2'], ind_inlet, ind_coflow)
        k4 = compute_all_k(Y_k4)

        # combine to final Y_new
        for name in Y:
            Y[name][2:-2, 2:-2] = (Y[name][2:-2, 2:-2] +
                                   dt/6.0 * (k1[name] + 2.0*k2[name] + 2.0*k3[name] + k4[name]))

        # enforce small-value clipping and BCs
        self.chemistry_boundaries_numba(Y['CH4'], Y['O2'], Y['CO2'], Y['H2O'], Y['N2'], ind_inlet, ind_coflow)
        for name in Y:
            Y[name][Y[name] < 1e-15] = 0.0
            self.chemistries[name].Y = Y[name]


# ==================== Numba-compiled functions ====================

@njit(parallel=True)
def _compute_reaction_rates_numba(T_field, Y_O2, Y_CH4, M_O2, M_CH4, T_A, A, rho, stoich_coef, molar_masses):
    """
    Compute reaction rates for all species using Arrhenius kinetics.
    
    Args:
        T_field: Temperature field
        Y_O2, Y_CH4: Mass fraction fields
        M_O2, M_CH4: Molar masses
        T_A: Activation temperature
        A: Pre-exponential factor
        rho: Density
        stoich_coef: Array [CH4, O2, CO2, H2O] stoichiometric coefficients
        molar_masses: Array [CH4, O2, CO2, H2O] molar masses
    
    Returns:
        Tuple of (k_CH4, k_O2, k_CO2, k_H2O)
    """
    # ===========================================================================
    # INITIALIZATION: Allocate arrays for reaction rates
    # ===========================================================================
    n, m = T_field.shape
    k_CH4 = np.zeros((n, m))
    k_O2 = np.zeros((n, m))
    k_CO2 = np.zeros((n, m))
    k_H2O = np.zeros((n, m))
    
    # ===========================================================================
    # REACTION RATE LOOP: Compute rates at each grid point
    # ===========================================================================
    for i in prange(n):
        for j in range(m):
            # ===================================================================
            # CONCENTRATIONS: Calculate molar concentrations
            # ===================================================================
            T = T_field[i, j]
            C_O2 = (Y_O2[i, j] / M_O2) * rho
            C_CH4 = (Y_CH4[i, j] / M_CH4) * rho
            
            # ===================================================================
            # ARRHENIUS RATE: Compute k = A * exp(-T_A/T) * [CH4] * [O2]^2
            # ===================================================================
            k = A * np.exp(-T_A / T) * (C_CH4) * (C_O2**2)
            
            # ===================================================================
            # SPECIES RATES: Apply stoichiometry to get individual rates
            # ===================================================================
            k_CH4[i, j] = stoich_coef[0] * molar_masses[0] * k
            k_O2[i, j] = stoich_coef[1] * molar_masses[1] * k
            k_CO2[i, j] = stoich_coef[2] * molar_masses[2] * k
            k_H2O[i, j] = stoich_coef[3] * molar_masses[3] * k
    
    return (k_CH4, k_O2, k_CO2, k_H2O)


@njit(parallel=True)
def _compute_heat_release_numba(rates_array, delta_h_array, molar_mass_array):
    """Numba-compiled function to compute heat release in parallel."""
    # ===========================================================================
    # INITIALIZATION: Setup heat release array
    # ===========================================================================
    n_species = len(rates_array)
    omega_T = np.zeros_like(rates_array[0])
    
    # ===========================================================================
    # HEAT RELEASE: Sum contributions from all species
    # ===========================================================================
    for i in prange(n_species):
        omega_T -= rates_array[i] * delta_h_array[i] / molar_mass_array[i]
    
    return omega_T


@njit(parallel=True)
def _apply_chemistry_boundaries_numba(Y_CH4, Y_O2, Y_CO2, Y_H2O, Y_N2, ind_inlet, ind_coflow, n, m):
    """Numba-compiled function to apply boundary conditions in parallel."""
    # ===========================================================================
    # LEFT BOUNDARY: Neumann condition (zero gradient)
    # ===========================================================================
    for j in prange(2, m-2):
        Y_CH4[0, j] = Y_CH4[2, j]
        Y_O2[0, j] = Y_O2[2, j]
        Y_CO2[0, j] = Y_CO2[2, j]
        Y_H2O[0, j] = Y_H2O[2, j]
        Y_N2[0, j] = Y_N2[2, j]
        Y_CH4[1, j] = Y_CH4[2, j]
        Y_O2[1, j] = Y_O2[2, j]
        Y_CO2[1, j] = Y_CO2[2, j]
        Y_H2O[1, j] = Y_H2O[2, j]
        Y_N2[1, j] = Y_N2[2, j]
    
    # ===========================================================================
    # INLET CH4: Slot region bottom wall (pure CH4)
    # ===========================================================================
    for i in prange(ind_inlet):
        Y_CH4[i, 0] = 1.0
        Y_O2[i, 0] = 0.0
        Y_CO2[i, 0] = 0.0
        Y_H2O[i, 0] = 0.0
        Y_N2[i, 0] = 0.0
        Y_CH4[i, 1] = 1.0
        Y_O2[i, 1] = 0.0
        Y_CO2[i, 1] = 0.0
        Y_H2O[i, 1] = 0.0
        Y_N2[i, 1] = 0.0
    
    # ===========================================================================
    # INLET O2+N2: Slot region top wall (air composition)
    # ===========================================================================
    for i in prange(ind_inlet):
        Y_CH4[i, m-1] = 0.0
        Y_O2[i, m-1] = 0.21
        Y_CO2[i, m-1] = 0.0
        Y_H2O[i, m-1] = 0.0
        Y_N2[i, m-1] = 0.79
        Y_CH4[i, m-2] = 0.0
        Y_O2[i, m-2] = 0.21
        Y_CO2[i, m-2] = 0.0
        Y_H2O[i, m-2] = 0.0
        Y_N2[i, m-2] = 0.79
    
    # ===========================================================================
    # COFLOW N2: Coflow region bottom wall (pure N2)
    # ===========================================================================
    for i in prange(ind_inlet, ind_coflow):
        Y_CH4[i, 0] = 0.0
        Y_O2[i, 0] = 0.0
        Y_CO2[i, 0] = 0.0
        Y_H2O[i, 0] = 0.0
        Y_N2[i, 0] = 1.0
        Y_CH4[i, 1] = 0.0
        Y_O2[i, 1] = 0.0
        Y_CO2[i, 1] = 0.0
        Y_H2O[i, 1] = 0.0
        Y_N2[i, 1] = 1.0
    
    # ===========================================================================
    # COFLOW N2: Coflow region top wall (pure N2)
    # ===========================================================================
    for i in prange(ind_inlet, ind_coflow):
        Y_CH4[i, m-1] = 0.0
        Y_O2[i, m-1] = 0.0
        Y_CO2[i, m-1] = 0.0
        Y_H2O[i, m-1] = 0.0
        Y_N2[i, m-1] = 1.0
        Y_CH4[i, m-2] = 0.0
        Y_O2[i, m-2] = 0.0
        Y_CO2[i, m-2] = 0.0
        Y_H2O[i, m-2] = 0.0
        Y_N2[i, m-2] = 1.0
    
    # ===========================================================================
    # WALL BOUNDARY: Lower wall outlet region (Neumann condition)
    # ===========================================================================
    for i in prange(ind_coflow, n):
        Y_CH4[i, 0] = Y_CH4[i, 2]
        Y_O2[i, 0] = Y_O2[i, 2]
        Y_CO2[i, 0] = Y_CO2[i, 2]
        Y_H2O[i, 0] = Y_H2O[i, 2]
        Y_N2[i, 0] = Y_N2[i, 2]
        Y_CH4[i, 1] = Y_CH4[i, 2]
        Y_O2[i, 1] = Y_O2[i, 2]
        Y_CO2[i, 1] = Y_CO2[i, 2]
        Y_H2O[i, 1] = Y_H2O[i, 2]
        Y_N2[i, 1] = Y_N2[i, 2]
    
    # ===========================================================================
    # WALL BOUNDARY: Upper wall outlet region (Neumann condition)
    # ===========================================================================
    for i in prange(ind_coflow, n):
        Y_CH4[i, m-1] = Y_CH4[i, m-3]
        Y_O2[i, m-1] = Y_O2[i, m-3]
        Y_CO2[i, m-1] = Y_CO2[i, m-3]
        Y_H2O[i, m-1] = Y_H2O[i, m-3]
        Y_N2[i, m-1] = Y_N2[i, m-3]
        Y_CH4[i, m-2] = Y_CH4[i, m-3]
        Y_O2[i, m-2] = Y_O2[i, m-3]
        Y_CO2[i, m-2] = Y_CO2[i, m-3]
        Y_H2O[i, m-2] = Y_H2O[i, m-3]
        Y_N2[i, m-2] = Y_N2[i, m-3]
    
    # ===========================================================================
    # OUTLET BOUNDARY: Right boundary (extrapolation)
    # ===========================================================================
    for j in prange(2, m-2):
        Y_CH4[n-1, j] = Y_CH4[n-3, j]
        Y_O2[n-1, j] = Y_O2[n-3, j]
        Y_CO2[n-1, j] = Y_CO2[n-3, j]
        Y_H2O[n-1, j] = Y_H2O[n-3, j]
        Y_N2[n-1, j] = Y_N2[n-3, j]
        Y_CH4[n-2, j] = Y_CH4[n-3, j]
        Y_O2[n-2, j] = Y_O2[n-3, j]
        Y_CO2[n-2, j] = Y_CO2[n-3, j]
        Y_H2O[n-2, j] = Y_H2O[n-3, j]
        Y_N2[n-2, j] = Y_N2[n-3, j]


@njit(parallel=True)
def _compute_species_rhs_numba(phi, u, v, diffusion_coef, source, inv_dx, inv_dy, rhs):
    """Numba-compiled function to compute species RHS in parallel."""
    # ===========================================================================
    # GRID SETUP: Extract dimensions and precompute inverses
    # ===========================================================================
    n_interior = rhs.shape[0]
    m_interior = rhs.shape[1]
    inv_dx2 = inv_dx * inv_dx
    inv_dy2 = inv_dy * inv_dy
    
    # ===========================================================================
    # RHS COMPUTATION LOOP: Compute right-hand side for each interior point
    # ===========================================================================
    for i in prange(n_interior):
        for j in range(m_interior):
            # ===================================================================
            # INDEX OFFSET: Map interior to actual grid position
            # ===================================================================
            ii = i + 2  # offset to actual grid position
            jj = j + 2
            
            # ===================================================================
            # FIELD EXTRACTION: Get local and neighbor values
            # ===================================================================
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
            
            # ===================================================================
            # ADVECTION: Upwind scheme for convective terms
            # ===================================================================
            if u_loc > 0:
                adv_x = u_loc * (phi_loc - phi_m1_x) * inv_dx
            else:
                adv_x = u_loc * (phi_p1_x - phi_loc) * inv_dx
            
            if v_loc > 0:
                adv_y = v_loc * (phi_loc - phi_m1_y) * inv_dy
            else:
                adv_y = v_loc * (phi_p1_y - phi_loc) * inv_dy
            
            # ===================================================================
            # DIFFUSION: 4th-order central differences
            # ===================================================================
            diff_x = (-phi_p2_x + 16.0*phi_p1_x - 30.0*phi_loc + 16.0*phi_m1_x - phi_m2_x) * inv_dx2 / 12.0
            diff_y = (-phi_p2_y + 16.0*phi_p1_y - 30.0*phi_loc + 16.0*phi_m1_y - phi_m2_y) * inv_dy2 / 12.0
            
            diff = diffusion_coef * (diff_x + diff_y)
            
            # ===================================================================
            # SOURCE TERM: Add chemical reaction contribution
            # ===================================================================
            src = source[ii, jj]
            
            # ===================================================================
            # COMBINE: Assemble total RHS
            # ===================================================================
            rhs[i, j] = -adv_x - adv_y + diff + src