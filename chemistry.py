import numpy as np
import constant as const

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

    def initiate_steps(self, dx, dy):
        self.inv_dx = 1/dx
        self.inv_dy = 1/dy

    def update_reaction_rates(self, T_field):
        """Compute Arrhenius-based reaction rates for the global reaction (same as before)."""
        O2 = self.chemistries['O2']
        CH4 = self.chemistries['CH4']
        exp_arg = -const.T_A / T_field
        k = const.A * np.exp(exp_arg)
        O2_conc = O2.Y * const.rho / O2.molar_mass
        CH4_conc = CH4.Y * const.rho / CH4.molar_mass
        Q = k * O2_conc**2 * CH4_conc
        reaction_rates = {}
        for name, chem in self.chemistries.items():
            nu_i = const.stoichiometric_coefficients.get(name, 0)
            rate = nu_i * Q * const.molar_mass[name]
            reaction_rates[name] = rate
        self.reaction_rates = reaction_rates
        return reaction_rates

    def heat_release(self):
        omega_T = 0.0
        for name, rate in self.reaction_rates.items():
            delta_h = const.enthalpy_of_formation.get(name, 0)
            omega_T -= rate * delta_h / self.chemistries[name].molar_mass
        return omega_T

    # Boundary conditions for species (mêmes BCs que dans System)
    def chemistry_boundaries(self, Y_CH4_new, Y_O2_new, Y_CO2_new, Y_H2O_new, Y_N2_new, ind_inlet, ind_coflow):
        # --- Left boundary (i=0,1) - Neumann (zero gradient) for 4th-order ---
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
        
        # --- CH4 inlet (slot region, bottom wall j=0,1) ---
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
        
        # --- O2+N2 inlet (slot region, top wall j=m-1, m-2) ---
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
        
        # --- N2 coflow inlet (coflow region, bottom wall j=0,1) ---
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
        
        # --- N2 coflow inlet (coflow region, top wall j=m-1,m-2) ---
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
        
        # --- Lower wall (outlet region, j=0,1) - Neumann ---
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
        
        # --- Upper wall (outlet region, j=m-1,m-2) - Neumann ---
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
        
        # --- Right boundary (outlet, i=n-1,n-2) - Extrapolation ---
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

    def rk4_advance_species(self, u, v, dt, ind_inlet, ind_coflow):
        # prepare arrays
        Y = {name: np.copy(chem.Y) for name, chem in self.chemistries.items()}
        # sources in mass-fraction units (kg/m3 -> divide by rho inside RHS call as source expects kg/kg/s)
        sources = {name: (self.reaction_rates[name] / const.rho) for name in self.chemistries.keys()}

        # helper to compute all k given current full-field arrays
        def compute_all_k(Y_fields):
            ks = {}
            for name, chem in self.chemistries.items():
                ks[name] = self.compute_species_rhs(Y_fields[name], u, v, chem.diffusivity, sources[name])
            return ks

        # k1
        k1 = compute_all_k(Y)

        # k2
        Y_k2 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k2[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + 0.5 * dt * k1[name]
        # apply BCs on temporary fields
        self.chemistry_boundaries(Y_k2['CH4'], Y_k2['O2'], Y_k2['CO2'], Y_k2['H2O'], Y_k2['N2'], ind_inlet, ind_coflow)
        k2 = compute_all_k(Y_k2)

        # k3
        Y_k3 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k3[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + 0.5 * dt * k2[name]
        self.chemistry_boundaries(Y_k3['CH4'], Y_k3['O2'], Y_k3['CO2'], Y_k3['H2O'], Y_k3['N2'], ind_inlet, ind_coflow)
        k3 = compute_all_k(Y_k3)

        # k4
        Y_k4 = {name: np.copy(Y[name]) for name in Y}
        for name in Y:
            Y_k4[name][2:-2, 2:-2] = Y[name][2:-2, 2:-2] + dt * k3[name]
        self.chemistry_boundaries(Y_k4['CH4'], Y_k4['O2'], Y_k4['CO2'], Y_k4['H2O'], Y_k4['N2'], ind_inlet, ind_coflow)
        k4 = compute_all_k(Y_k4)

        # combine to final Y_new (interior 4th-order region)
        for name in Y:
            Y[name][2:-2, 2:-2] = (Y[name][2:-2, 2:-2] +
                                   dt/6.0 * (k1[name] + 2.0*k2[name] + 2.0*k3[name] + k4[name]))

        # enforce small-value clipping and BCs
        self.chemistry_boundaries(Y['CH4'], Y['O2'], Y['CO2'], Y['H2O'], Y['N2'], ind_inlet, ind_coflow)
        for name in Y:
            Y[name][Y[name] < 1e-15] = 0.0
            self.chemistries[name].Y = Y[name]