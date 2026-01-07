import numpy as np
import constant as const
from numba import jit, prange

class Fluid:
    """Fluid class representing a 2D incompressible fluid solver data and operations.

    This class stores grid dimensions, physical properties and field arrays
    (velocity components u, v and pressure P). It provides routines for:
      - initializing velocity/pressure fields,
      - setting time step and grid spacing helpers,
      - performing advection-diffusion updates,
      - solving the pressure Poisson equation using SOR,
      - applying boundary conditions and correcting velocities.

    Attributes:
        n (int): number of grid points in the x-direction (including boundaries).
        m (int): number of grid points in the y-direction (including boundaries).
        diffusivity (float): kinematic diffusivity / viscosity used in diffusion terms.
        u, v, P (ndarray): velocity and pressure fields (initialized later).
        dt, inv_dx, inv_dy (float): time step and inverse grid spacings (set later).
    """
    
    def __init__(self, n, m, diffusivity, rho):
        """Initialize basic solver parameters.

        Args:
            n (int): grid size in x-direction (including boundary nodes).
            m (int): grid size in y-direction (including boundary nodes).
            diffusivity (float): diffusion coefficient (e.g. kinematic viscosity).

        This constructor only stores basic geometry and physical parameter(s).
        The actual field arrays (u, v, P) should be created via
        velocity_initialization(), and time/spacing values set via initiate_steps().
        """
        self.n = n
        self.m = m
        self.nu = diffusivity
        self.rho = rho
        self.i_slice = slice(2, -2)
        self.j_slice = slice(2, -2)

    def velocity_initialization(self, u_initial, v_initial, P_initial):
        """Initialize the velocity field."""
        self.u = np.array(u_initial)
        self.v = np.array(v_initial)
        self.P = np.array(P_initial)

    def initiate_steps(self, dt, dx, dy):
        self.dt = dt
        self.inv_dx = 1/dx
        self.inv_dy = 1/dy

    # Ajout : helper vectorisé pour advection-diffusion (utilisable aussi pour scalaires)
    def adv_diff_interior(self, inv_dx, inv_dy, src_u=None, src_v=None):
        u_new = np.copy(self.u)
        v_new = np.copy(self.v)
        
        # ============================================================================
        # INITIALIZATION: Extract local velocity fields
        # ============================================================================
        u_loc = self.u[self.i_slice, self.j_slice]
        v_loc = self.v[self.i_slice, self.j_slice]
        
        # ============================================================================
        # U-MOMENTUM: Lax-Wendroff half-step values computation
        # ============================================================================
        # X-direction half-steps for u
        uphalf_x = 0.5 * (self.u[self.i_slice, self.j_slice] + self.u[3:-1, self.j_slice]) - \
                   0.25 * self.dt * inv_dx * (self.u[3:-1, self.j_slice]**2 - self.u[self.i_slice, self.j_slice]**2)
        umhalf_x = 0.5 * (self.u[1:-3, self.j_slice] + self.u[self.i_slice, self.j_slice]) - \
                   0.25 * self.dt * inv_dx * (self.u[self.i_slice, self.j_slice]**2 - self.u[1:-3, self.j_slice]**2)
        
        # Y-direction half-steps for v (needed for cross-terms)
        vphalf_y = 0.5 * (self.v[self.i_slice, self.j_slice] + self.v[self.i_slice, 3:-1]) - \
                   0.25 * self.dt * inv_dy * (self.v[self.i_slice, 3:-1]**2 - self.v[self.i_slice, self.j_slice]**2)
        vmhalf_y = 0.5 * (self.v[self.i_slice, 1:-3] + self.v[self.i_slice, self.j_slice]) - \
                   0.25 * self.dt * inv_dy * (self.v[self.i_slice, self.j_slice]**2 - self.v[self.i_slice, 1:-3]**2)
        
        # Cross-direction half-steps for u in y-direction
        uphalf_y = 0.5 * (self.u[self.i_slice, self.j_slice] + self.u[self.i_slice, 3:-1]) - \
                   0.25 * self.dt * inv_dy * (self.v[self.i_slice, 3:-1]*self.u[self.i_slice, 3:-1] - 
                                              self.v[self.i_slice, self.j_slice]*self.u[self.i_slice, self.j_slice])
        umhalf_y = 0.5 * (self.u[self.i_slice, 1:-3] + self.u[self.i_slice, self.j_slice]) - \
                   0.25 * self.dt * inv_dy * (self.v[self.i_slice, self.j_slice]*self.u[self.i_slice, self.j_slice] - 
                                              self.v[self.i_slice, 1:-3]*self.u[self.i_slice, 1:-3])
        
        # ============================================================================
        # U-MOMENTUM: Advection terms (Lax-Wendroff scheme)
        # ============================================================================
        adv_x_u = 0.5 * inv_dx * (uphalf_x**2 - umhalf_x**2)
        adv_y_u = 0.5 * inv_dy * (vphalf_y * uphalf_y - vmhalf_y * umhalf_y)
        
        # ============================================================================
        # U-MOMENTUM: Diffusion terms (4th order accurate)
        # ============================================================================
        diff_x_u = (-self.u[4:, self.j_slice] + 16.0*self.u[3:-1, self.j_slice] - 30.0*u_loc + 
                    16.0*self.u[1:-3, self.j_slice] - self.u[:-4, self.j_slice]) * inv_dx**2 / 12.0
        diff_y_u = (-self.u[self.i_slice, 4:] + 16.0*self.u[self.i_slice, 3:-1] - 30.0*u_loc + 
                    16.0*self.u[self.i_slice, 1:-3] - self.u[self.i_slice, :-4]) * inv_dy**2 / 12.0
        diff_u = self.nu * (diff_x_u + diff_y_u)
        
        # ============================================================================
        # U-MOMENTUM: Time integration update
        # ============================================================================
        u_new[self.i_slice, self.j_slice] = u_loc + self.dt * (-adv_x_u - adv_y_u + diff_u)
        
        # ============================================================================
        # V-MOMENTUM: Lax-Wendroff half-step values computation
        # ============================================================================
        # Cross-direction half-steps for v in x-direction
        vphalh_x = 0.5 * (self.v[self.i_slice, self.j_slice] + self.v[3:-1, self.j_slice]) - \
                   0.25 * self.dt * inv_dx * (self.u[3:-1, self.j_slice]*self.v[3:-1, self.j_slice] - 
                                              self.u[self.i_slice, self.j_slice]*self.v[self.i_slice, self.j_slice])
        vmhalh_x = 0.5 * (self.v[1:-3, self.j_slice] + self.v[self.i_slice, self.j_slice]) - \
                   0.25 * self.dt * inv_dx * (self.u[self.i_slice, self.j_slice]*self.v[self.i_slice, self.j_slice] - 
                                              self.u[1:-3, self.j_slice]*self.v[1:-3, self.j_slice])
        
        # ============================================================================
        # V-MOMENTUM: Advection terms (Lax-Wendroff scheme)
        # ============================================================================
        adv_x_v = 0.5 * inv_dx * (uphalf_x * vphalh_x - umhalf_x * vmhalh_x)
        adv_y_v = 0.5 * inv_dy * (vphalf_y**2 - vmhalf_y**2)
        
        # ============================================================================
        # V-MOMENTUM: Diffusion terms (4th order accurate)
        # ============================================================================
        diff_x_v = (-self.v[4:, self.j_slice] + 16.0*self.v[3:-1, self.j_slice] - 30.0*v_loc + 
                    16.0*self.v[1:-3, self.j_slice] - self.v[:-4, self.j_slice]) * inv_dx**2 / 12.0
        diff_y_v = (-self.v[self.i_slice, 4:] + 16.0*self.v[self.i_slice, 3:-1] - 30.0*v_loc + 
                    16.0*self.v[self.i_slice, 1:-3] - self.v[self.i_slice, :-4]) * inv_dy**2 / 12.0
        diff_v = self.nu * (diff_x_v + diff_y_v)
        
        # ============================================================================
        # V-MOMENTUM: Time integration update
        # ============================================================================
        v_new[self.i_slice, self.j_slice] = v_loc + self.dt * (-adv_x_v - adv_y_v + diff_v)

        return u_new, v_new

    # Optionnel : méthode pour appliquer BCs de vitesse si on veut externaliser
    def apply_velocity_bcs(self, u_upd, v_upd, ind_inlet, ind_coflow, Uslot, Ucoflow):   
        """
        Apply velocity boundary conditions:
        - Inlet (left boundary): specified velocities for CH4 and O2+N2 inlets
        - Walls (top and bottom): no-slip (u=v=0)
        - Outlet (right boundary): Neumann (zero-gradient) for u and v
        Args:
            u_upd, v_upd: updated velocity fields to apply BCs on
            ind_inlet: index separating inlet slot region from coflow region
            ind_coflow: index separating coflow region from outlet region
            Uslot: inlet velocity magnitude for slot region
            Ucoflow: inlet velocity magnitude for coflow region
        """     
        # ============================================================================
        # INLET BOUNDARY: CH4 inlet (slot region, bottom wall)
        # ============================================================================
        u_upd[:ind_inlet, 0] = 0
        v_upd[:ind_inlet, 0] = Uslot
        u_upd[:ind_inlet, 1] = 0
        v_upd[:ind_inlet, 1] = Uslot
        
        # ============================================================================
        # INLET BOUNDARY: O2+N2 inlet (slot region, top wall)
        # ============================================================================
        u_upd[:ind_inlet, self.m-1] = 0
        v_upd[:ind_inlet, self.m-1] = -Uslot
        u_upd[:ind_inlet, self.m-2] = 0
        v_upd[:ind_inlet, self.m-2] = -Uslot
        
        # ============================================================================
        # COFLOW BOUNDARY: N2 coflow inlet (bottom wall)
        # ============================================================================
        u_upd[ind_inlet:ind_coflow, 0] = 0
        v_upd[ind_inlet:ind_coflow, 0] = Ucoflow
        u_upd[ind_inlet:ind_coflow, 1] = 0
        v_upd[ind_inlet:ind_coflow, 1] = Ucoflow
        
        # ============================================================================
        # COFLOW BOUNDARY: N2 coflow inlet (top wall)
        # ============================================================================
        u_upd[ind_inlet:ind_coflow, self.m-1] = 0
        v_upd[ind_inlet:ind_coflow, self.m-1] = -Ucoflow
        u_upd[ind_inlet:ind_coflow, self.m-2] = 0
        v_upd[ind_inlet:ind_coflow, self.m-2] = -Ucoflow
        
        # ============================================================================
        # WALL BOUNDARY: Lower wall (outlet region, no-slip)
        # ============================================================================
        u_upd[ind_coflow:, 0] = 0
        v_upd[ind_coflow:, 0] = 0
        u_upd[ind_coflow:, 1] = 0
        v_upd[ind_coflow:, 1] = 0
        
        # ============================================================================
        # WALL BOUNDARY: Upper wall (outlet region, no-slip)
        # ============================================================================
        u_upd[ind_coflow:, self.m-1] = 0
        v_upd[ind_coflow:, self.m-1] = 0
        u_upd[ind_coflow:, self.m-2] = 0
        v_upd[ind_coflow:, self.m-2] = 0
        
        # ============================================================================
        # OUTLET BOUNDARY: Right boundary (extrapolation/zero-gradient)
        # ============================================================================
        u_upd[self.n-1, 2:self.m-2] = u_upd[self.n-3, 2:self.m-2]
        v_upd[self.n-1, 2:self.m-2] = v_upd[self.n-3, 2:self.m-2]
        u_upd[self.n-2, 2:self.m-2] = u_upd[self.n-3, 2:self.m-2]
        v_upd[self.n-2, 2:self.m-2] = v_upd[self.n-3, 2:self.m-2]
        
    def SOR_pressure_solver(self, u, v):
        """
        Solve pressure Poisson equation for incompressible flow:
        ∇²p = (ρ/Δt) ∇·u*
        where u* is the intermediate velocity field
        """
        # ============================================================================
        # DIVERGENCE COMPUTATION: Calculate velocity divergence (∇·u*)
        # ============================================================================
        
        # Central differences for divergence
        du_dx = 0.5 * self.inv_dx * (u[3:-1, self.j_slice] - u[1:-3, self.j_slice])
        dv_dy = 0.5 * self.inv_dy * (v[self.i_slice, 3:-1] - v[self.i_slice, 1:-3])
        
        # ============================================================================
        # RHS CONSTRUCTION: Build right-hand side (ρ/Δt)∇·u*
        # ============================================================================
        f = np.zeros_like(self.P)
        f[self.i_slice, self.j_slice] = (self.rho / self.dt) * (du_dx + dv_dy)

        # ============================================================================
        # SOR SOLVER INITIALIZATION: Setup parameters and arrays
        # ============================================================================
        P_new = np.copy(self.P)
        omega = 1.5
        tolerance = 1e-6
        max_iterations = 2000

        f_in = f[self.i_slice, self.j_slice]
        denom = 2.0 * (self.inv_dx**2 + self.inv_dy**2)

        # ============================================================================
        # RED-BLACK ORDERING: Setup checkerboard pattern for SOR
        # ============================================================================
        ii, jj = np.indices((self.n-4, self.m-4))
        mask_red = ((ii + jj) % 2) == 0
        mask_black = ~mask_red

        def compute_Pgs_local(P):
            P_ip = P[3:-1, self.j_slice]
            P_im = P[1:-3, self.j_slice]
            P_jp = P[self.i_slice, 3:-1]
            P_jm = P[self.i_slice, 1:-3]
            laplacian = (P_ip + P_im) * self.inv_dx**2 + (P_jp + P_jm) * self.inv_dy**2
            return (laplacian - f_in) / denom

        # ============================================================================
        # SOR ITERATION: Solve ∇²P = f using successive over-relaxation
        # ============================================================================
        for iteration in range(max_iterations):
            P_old = P_new.copy()
            P_in = P_new[self.i_slice, self.j_slice]

            # Update red points
            P_gs = compute_Pgs_local(P_new)
            P_in[mask_red] = (1.0 - omega) * P_in[mask_red] + omega * P_gs[mask_red]

            # Update black points
            P_gs = compute_Pgs_local(P_new)
            P_in[mask_black] = (1.0 - omega) * P_in[mask_black] + omega * P_gs[mask_black]

            # Check convergence
            residual = np.max(np.abs(P_new - P_old))
            if residual < tolerance:
                break

        # ============================================================================
        # PRESSURE BOUNDARY CONDITIONS: Apply Neumann and Dirichlet conditions
        # ============================================================================
        # Left (inlet): ∂P/∂x = 0
        P_new[0, :] = P_new[2, :]
        P_new[1, :] = P_new[2, :]
        # Right (outlet): P = 0 (reference)
        P_new[-1, :] = 0
        P_new[-2, :] = 0
        # Walls: ∂P/∂n = 0
        P_new[:, 0] = P_new[:, 2]
        P_new[:, 1] = P_new[:, 2]
        P_new[:, -1] = P_new[:, -3]
        P_new[:, -2] = P_new[:, -3]

        self.P = P_new
    
    def correction_velocity(self, u_star, v_star, dt, inv_dx, inv_dy, n, m):
        """
        Correct intermediate velocity: u^(n+1) = u* - (Δt/ρ)∇P
        Cette correction REMPLACE u* et v*, pas s'ajoute à un calcul précédent
        """
        
        # ============================================================================
        # PRESSURE GRADIENT: Compute ∇P using central differences
        # ============================================================================
        dp_dx = (self.P[3:-1, self.j_slice] - self.P[1:-3, self.j_slice]) * self.inv_dx * 0.5
        dp_dy = (self.P[self.i_slice, 3:-1] - self.P[self.i_slice, 1:-3]) * self.inv_dy * 0.5

        # ============================================================================
        # VELOCITY CORRECTION: Apply pressure gradient to obtain divergence-free field
        # ============================================================================
        u_new = np.copy(u_star)
        v_new = np.copy(v_star)
        
        u_new[self.i_slice, self.j_slice] = u_star[self.i_slice, self.j_slice] - (self.dt / self.rho) * dp_dx
        v_new[self.i_slice, self.j_slice] = v_star[self.i_slice, self.j_slice] - (self.dt / self.rho) * dp_dy

        # ======================================================================
        # LEFT BOUNDARY: Special handling for inlet boundary
        # ======================================================================
        for i in range(2):
            for j in range(2, m-2):
                v_loc = v_star[i, j]
                u_new[i, j] = 0.0
                
                # Upwind advection for v
                if v_loc >= 0:
                    adv_v_y = v_loc * (v_loc - v_star[i, j-1]) * inv_dy
                else:
                    adv_v_y = v_loc * (v_star[i, j+1] - v_loc) * inv_dy
                
                # Diffusion for v at boundary
                diffusion_v = self.nu * (
                    (2*v_star[i+1, j] - 2*v_loc) * inv_dx**2 +
                    (v_star[i, j+1] - 2*v_loc + v_star[i, j-1]) * inv_dy**2
                )
                
                # Pressure gradient at boundary
                dp_dy_local = (self.P[2, j] - self.P[0, j]) * inv_dy * 0.5
                v_new[i, j] = v_loc + dt * (-adv_v_y + diffusion_v - (dt / self.rho) * dp_dy_local)

        return u_new, v_new

    # ==================== Numba-accelerated versions ====================

    @staticmethod
    @jit(nopython=True, parallel=True)
    def _adv_diff_kernel(u, v, dt, inv_dx, inv_dy, diffusivity, n, m):
        """Numba-optimized kernel for advection-diffusion computation."""
        u_new = np.copy(u)
        v_new = np.copy(v)
        
        for i in prange(2, n-2):
            for j in range(2, m-2):
                # ==================================================================
                # LOCAL VALUES: Extract current cell velocities
                # ==================================================================
                u_loc = u[i, j]
                v_loc = v[i, j]

                # ==================================================================
                # LAX-WENDROFF HALF-STEPS: Compute intermediate values
                # ==================================================================
                uphalf_x = 0.5 * (u[i, j] + u[i+1, j]) - 0.25 * dt * inv_dx * (u[i+1, j]**2 - u[i, j]**2)
                umhalf_x = 0.5 * (u[i-1, j] + u[i, j]) - 0.25 * dt * inv_dx * (u[i, j]**2 - u[i-1, j]**2)
                vphalf_y = 0.5 * (v[i, j] + v[i, j+1]) - 0.25 * dt * inv_dy * (v[i, j+1]**2 - v[i, j]**2)
                vmhalf_y = 0.5 * (v[i, j-1] + v[i, j]) - 0.25 * dt * inv_dy * (v[i, j]**2 - v[i, j-1]**2)
                
                vphalh_x = 0.5 * (v[i, j] + v[i+1, j]) - 0.25 * dt * inv_dx * (u[i+1, j]*v[i+1, j] - u[i, j]*v[i, j])
                vmhalh_x = 0.5 * (v[i-1, j] + v[i, j]) - 0.25 * dt * inv_dx * (u[i, j]*v[i, j] - u[i-1, j]*v[i-1, j])
                uphalf_y = 0.5 * (u[i, j] + u[i, j+1]) - 0.25 * dt * inv_dy * (v[i, j+1]*u[i, j+1] - v[i, j]*u[i, j])
                umhalf_y = 0.5 * (u[i, j-1] + u[i, j]) - 0.25 * dt * inv_dy * (v[i, j]*u[i, j] - v[i, j-1]*u[i, j-1])

                # ==================================================================
                # ADVECTION TERMS: Compute using Lax-Wendroff scheme
                # ==================================================================
                adv_x_u = 0.5 * inv_dx * (uphalf_x**2 - umhalf_x**2)
                adv_y_u = 0.5 * inv_dy * (vphalf_y * uphalf_y - vmhalf_y * umhalf_y)

                adv_x_v = 0.5 * inv_dx * (uphalf_x * vphalh_x - umhalf_x * vmhalh_x)
                adv_y_v = 0.5 * inv_dy * (vphalf_y**2 - vmhalf_y**2)

                # ==================================================================
                # DIFFUSION U: Compute 4th-order accurate diffusion for u
                # ==================================================================
                diff_x_u = (-u[i+2, j] + 16.0*u[i+1, j] - 30.0*u_loc + 
                           16.0*u[i-1, j] - u[i-2, j]) * inv_dx**2 / 12.0
                diff_y_u = (-u[i, j+2] + 16.0*u[i, j+1] - 30.0*u_loc + 
                           16.0*u[i, j-1] - u[i, j-2]) * inv_dy**2 / 12.0
                diff_u = diffusivity * (diff_x_u + diff_y_u)
                
                # ==================================================================
                # UPDATE U: Time integration for u-momentum
                # ==================================================================
                u_new[i, j] = u_loc + dt * (-adv_x_u - adv_y_u + diff_u)
                
                # ==================================================================
                # DIFFUSION V: Compute 4th-order accurate diffusion for v
                # ==================================================================
                diff_x_v = (-v[i+2, j] + 16.0*v[i+1, j] - 30.0*v_loc + 
                           16.0*v[i-1, j] - v[i-2, j]) * inv_dx**2 / 12.0
                diff_y_v = (-v[i, j+2] + 16.0*v[i, j+1] - 30.0*v_loc + 
                           16.0*v[i, j-1] - v[i, j-2]) * inv_dy**2 / 12.0
                diff_v = diffusivity * (diff_x_v + diff_y_v)
                
                # ==================================================================
                # UPDATE V: Time integration for v-momentum
                # ==================================================================
                v_new[i, j] = v_loc + dt * (-adv_x_v - adv_y_v + diff_v)
        
        return u_new, v_new
    
    def adv_diff_interior_numba(self, inv_dx, inv_dy, src_u=None, src_v=None):
        """Numba-accelerated version of advection-diffusion computation."""
        return self._adv_diff_kernel(self.u, self.v, self.dt, inv_dx, inv_dy, 
                                     self.nu, self.n, self.m)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _sor_kernel(P, u, v, inv_dx, inv_dy, dt, omega, tolerance, max_iterations, n, m):
        """Numba SOR kernel pour fluide incompressible."""
        P_new = np.copy(P)
        
        # ======================================================================
        # RHS COMPUTATION: Calculate (ρ/Δt) * ∇·u
        # ======================================================================
        f = np.zeros_like(P)
        for i in prange(2, n-2):
            for j in range(2, m-2):
                du_dx = 0.5 * inv_dx * (u[i+1, j] - u[i-1, j])
                dv_dy = 0.5 * inv_dy * (v[i, j+1] - v[i, j-1])
                f[i, j] = (const.rho / dt) * (du_dx + dv_dy)
        
        denom = 2.0 * (inv_dx**2 + inv_dy**2)
        
        # ======================================================================
        # SOR ITERATION: Solve Poisson equation with red-black ordering
        # ======================================================================
        for iteration in range(max_iterations):
            P_old = P_new.copy()
            
            # Update red points (even i+j)
            for i in prange(2, n-2):
                for j in range(2, m-2):
                    if (i + j) % 2 == 0:
                        laplacian = ((P_new[i+1, j] + P_new[i-1, j]) * inv_dx**2 +
                                    (P_new[i, j+1] + P_new[i, j-1]) * inv_dy**2)
                        P_gs = (laplacian - f[i, j]) / denom
                        P_new[i, j] = (1.0 - omega) * P_new[i, j] + omega * P_gs
            
            # Update black points (odd i+j)
            for i in prange(2, n-2):
                for j in range(2, m-2):
                    if (i + j) % 2 == 1:
                        laplacian = ((P_new[i+1, j] + P_new[i-1, j]) * inv_dx**2 +
                                    (P_new[i, j+1] + P_new[i, j-1]) * inv_dy**2)
                        P_gs = (laplacian - f[i, j]) / denom
                        P_new[i, j] = (1.0 - omega) * P_new[i, j] + omega * P_gs
            
            # Check convergence
            residual = 0.0
            for i in range(n):
                for j in range(m):
                    diff = abs(P_new[i, j] - P_old[i, j])
                    if diff > residual:
                        residual = diff
            
                if residual < tolerance:
                    break
        
        # ======================================================================
        # BOUNDARY CONDITIONS: Apply pressure boundary conditions
        # ======================================================================
        P_new[0, :] = P_new[2, :]
        P_new[1, :] = P_new[2, :]
        P_new[-1, :] = 0.0
        P_new[-2, :] = 0.0
        P_new[:, 0] = P_new[:, 2]
        P_new[:, 1] = P_new[:, 2]
        P_new[:, -1] = P_new[:, -3]
        P_new[:, -2] = P_new[:, -3]
        
        return P_new
    
    def SOR_pressure_solver_numba(self, u, v):
        self.P = self._sor_kernel(self.P, u, v, self.inv_dx, self.inv_dy,
                                  self.dt, 1.5, 1e-6, 2000, 
                                self.n, self.m)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _correction_kernel(u_star, v_star, P, dt, inv_dx, inv_dy, rho, nu, n, m):
        """Numba-optimized velocity correction kernel."""
        u_new = np.copy(u_star)
        v_new = np.copy(v_star)
        
        # ======================================================================
        # INTERIOR CORRECTION: Apply pressure gradient correction
        # ======================================================================
        for i in prange(2, n-2):
            for j in range(2, m-2):
                dp_dx = (P[i+1, j] - P[i-1, j]) * inv_dx * 0.5
                dp_dy = (P[i, j+1] - P[i, j-1]) * inv_dy * 0.5
                
                u_new[i, j] = u_star[i, j] - (dt / rho) * dp_dx
                v_new[i, j] = v_star[i, j] - (dt / rho) * dp_dy
        
        # ======================================================================
        # LEFT BOUNDARY: Special handling for inlet boundary
        # ======================================================================
        for i in range(2):
            for j in range(2, m-2):
                v_loc = v_star[i, j]
                u_new[i, j] = 0.0
                
                # Upwind advection for v
                if v_loc >= 0:
                    adv_v_y = v_loc * (v_loc - v_star[i, j-1]) * inv_dy
                else:
                    adv_v_y = v_loc * (v_star[i, j+1] - v_loc) * inv_dy
                
                # Diffusion for v at boundary
                diffusion_v = nu * (
                    (2*v_star[i+1, j] - 2*v_loc) * inv_dx**2 +
                    (v_star[i, j+1] - 2*v_loc + v_star[i, j-1]) * inv_dy**2
                )
                
                # Pressure gradient at boundary
                dp_dy_local = (P[2, j] - P[0, j]) * inv_dy * 0.5
                v_new[i, j] = v_loc + dt * (-adv_v_y + diffusion_v - (dt / rho) * dp_dy_local)
        
        return u_new, v_new
    
    def correction_velocity_numba(self, u_star, v_star):
        """Numba-accelerated version of velocity correction."""
        return self._correction_kernel(u_star, v_star, self.P, self.dt, 
                                      self.inv_dx, self.inv_dy, const.rho, 
                                      const.nu, self.n, self.m)