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
    
    def __init__(self, n, m, diffusivity):
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
        self.diffusivity = diffusivity

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
        
        i_slice = slice(2, -2); j_slice = slice(2, -2)
        u_loc = self.u[i_slice, j_slice]; v_loc = self.v[i_slice, j_slice]
        u_pos = np.maximum(u_loc, 0); u_neg = np.minimum(u_loc, 0)
        v_pos = np.maximum(v_loc, 0); v_neg = np.minimum(v_loc, 0)
        
        # shifts in x
        u_m2_x = self.u[:-4, j_slice]
        u_m1_x = self.u[1:-3, j_slice]
        u_p1_x = self.u[3:-1, j_slice]
        u_p2_x = self.u[4:, j_slice]
        # shifts in y
        u_m2_y = self.u[i_slice, :-4]
        u_m1_y = self.u[i_slice, 1:-3]
        u_p1_y = self.u[i_slice, 3:-1]
        u_p2_y = self.u[i_slice, 4:]

        # Update u component
        adv_x_u = (u_pos * (u_loc - u_m1_x) * inv_dx +
                    u_neg * (u_p1_x - u_loc) * inv_dx)
        adv_y_u = (v_pos * (u_loc - u_m1_y) * inv_dy +
                    v_neg * (u_p1_y - u_loc) * inv_dy)
        diff_x_u = (-u_p2_x + 16.0*u_p1_x - 30.0*u_loc + 16.0*u_m1_x - u_m2_x) * inv_dx**2 / 12.0
        diff_y_u = (-u_p2_y + 16.0*u_p1_y - 30.0*u_loc + 16.0*u_m1_y - u_m2_y) * inv_dy**2 / 12.0
        diff_u = self.diffusivity * (diff_x_u + diff_y_u)
        # src_term_u = src_u[i_slice, j_slice] if src_u is not None else 0.0
        u_new[i_slice, j_slice] = u_loc + self.dt * (-adv_x_u - adv_y_u + diff_u )

        # shifts in x
        v_m2_x = self.v[:-4, j_slice]
        v_m1_x = self.v[1:-3, j_slice]
        v_p1_x = self.v[3:-1, j_slice]
        v_p2_x = self.v[4:, j_slice]
        # shifts in y
        v_m2_y = self.v[i_slice, :-4]
        v_m1_y = self.v[i_slice, 1:-3]
        v_p1_y = self.v[i_slice, 3:-1]
        v_p2_y = self.v[i_slice, 4:]

        # Update v component
        adv_x_v = (u_pos * (v_loc - v_m1_x) * inv_dx +
                    u_neg * (v_p1_x - v_loc) * inv_dx)
        adv_y_v = (v_pos * (v_loc - v_m1_y) * inv_dy +
                    v_neg * (v_p1_y - v_loc) * inv_dy)
        diff_x_v = (-v_p2_x + 16.0*v_p1_x - 30.0*v_loc + 16.0*v_m1_x - v_m2_x) * inv_dx**2 / 12.0
        diff_y_v = (-v_p2_y + 16.0*v_p1_y - 30.0*v_loc + 16.0*v_m1_y - v_m2_y) * inv_dy**2 / 12.0
        diff_v = self.diffusivity * (diff_x_v + diff_y_v)
        # src_term_v = src_v[i_slice, j_slice] if src_v is not None else 0.0
        v_new[i_slice, j_slice] = v_loc + self.dt * (-adv_x_v - adv_y_v + diff_v)

        return u_new, v_new

    # Optionnel : méthode pour appliquer BCs de vitesse si on veut externaliser
    def apply_velocity_bcs(self, u_upd, v_upd, ind_inlet, ind_coflow, Uslot, Ucoflow):        
        # CH4 inlet (slot region, bottom wall)
        u_upd[:ind_inlet, 0] = 0
        v_upd[:ind_inlet, 0] = Uslot
        u_upd[:ind_inlet, 1] = 0
        v_upd[:ind_inlet, 1] = Uslot
        
        # O2+N2 inlet (slot region, top wall)
        u_upd[:ind_inlet, self.m-1] = 0
        v_upd[:ind_inlet, self.m-1] = -Uslot
        u_upd[:ind_inlet, self.m-2] = 0
        v_upd[:ind_inlet, self.m-2] = -Uslot
        
        # N2 coflow inlet (coflow region, bottom wall)
        u_upd[ind_inlet:ind_coflow, 0] = 0
        v_upd[ind_inlet:ind_coflow, 0] = Ucoflow
        u_upd[ind_inlet:ind_coflow, 1] = 0
        v_upd[ind_inlet:ind_coflow, 1] = Ucoflow
        
        # N2 coflow inlet (coflow region, top wall)
        u_upd[ind_inlet:ind_coflow, self.m-1] = 0
        v_upd[ind_inlet:ind_coflow, self.m-1] = -Ucoflow
        u_upd[ind_inlet:ind_coflow, self.m-2] = 0
        v_upd[ind_inlet:ind_coflow, self.m-2] = -Ucoflow
        
        # Lower wall (outlet region)
        u_upd[ind_coflow:, 0] = 0
        v_upd[ind_coflow:, 0] = 0
        u_upd[ind_coflow:, 1] = 0
        v_upd[ind_coflow:, 1] = 0
        
        # Upper wall (outlet region)
        u_upd[ind_coflow:, self.m-1] = 0
        v_upd[ind_coflow:, self.m-1] = 0
        u_upd[ind_coflow:, self.m-2] = 0
        v_upd[ind_coflow:, self.m-2] = 0
        
        # Right boundary (outlet - extrapolation)
        u_upd[self.n-1, 2:self.m-2] = u_upd[self.n-3, 2:self.m-2]
        v_upd[self.n-1, 2:self.m-2] = v_upd[self.n-3, 2:self.m-2]
        u_upd[self.n-2, 2:self.m-2] = u_upd[self.n-3, 2:self.m-2]
        v_upd[self.n-2, 2:self.m-2] = v_upd[self.n-3, 2:self.m-2]
        
    def SOR_pressure_solver(self, u, v):
        """
        Solve pressure Poisson equation using SOR for:
        ∇²p = -ρ [ (∂u/∂x)^2 + 2 (∂u/∂y)(∂v/∂x) + (∂v/∂y)^2 ]
        Uses central differences for derivatives and red-black SOR.
        Args:
            u, v: velocity fields (same grid as self.P)
        """
        # --- Compute derivatives with central differences on interior [2:-2, 2:-2] ---
        i_slice = slice(2, -2)
        j_slice = slice(2, -2)
        
        du_dx = 0.5 * self.inv_dx * (u[3:-1, j_slice] - u[1:-3, j_slice])
        du_dy = 0.5 * self.inv_dy * (u[i_slice, 3:-1] - u[i_slice, 1:-3])
        dv_dx = 0.5 * self.inv_dx * (v[3:-1, j_slice] - v[1:-3, j_slice])
        dv_dy = 0.5 * self.inv_dy * (v[i_slice, 3:-1] - v[i_slice, 1:-3])

        # RHS: -rho * [ (du_dx)^2 + 2*(du_dy)*(dv_dx) + (dv_dy)^2 ]
        f = np.zeros_like(self.P)
        f_interior = -const.rho * (du_dx**2 + 2.0 * du_dy * dv_dx + dv_dy**2)
        f[i_slice, j_slice] = f_interior

        # === Initialize SOR ===
        P_new = np.copy(self.P)
        omega = 1.5
        tolerance = 1e-6
        max_iterations = 2000

        f_in = f[i_slice, j_slice]
        denom = 2.0 * (self.inv_dx**2 + self.inv_dy**2)

        # Precompute index masks for red-black pattern (interior is now n-4 x m-4)
        ii, jj = np.indices((self.n-4, self.m-4))
        mask_red = ((ii + jj) % 2) == 0
        mask_black = ~mask_red

        def compute_Pgs_local(P):
            P_ip = P[3:-1, j_slice]     # P(i+1, j)
            P_im = P[1:-3, j_slice]     # P(i-1, j)
            P_jp = P[i_slice, 3:-1]     # P(i, j+1)
            P_jm = P[i_slice, 1:-3]     # P(i, j-1)
            laplacian = (P_ip + P_im) * self.inv_dx**2 + (P_jp + P_jm) * self.inv_dy**2
            return (laplacian - f_in) / denom

        for iteration in range(max_iterations):
            P_old = P_new.copy()

            P_in = P_new[i_slice, j_slice]

            # Update red points
            P_gs = compute_Pgs_local(P_new)
            P_in[mask_red] = (1.0 - omega) * P_in[mask_red] + omega * P_gs[mask_red]

            # Update black points
            P_gs = compute_Pgs_local(P_new)
            P_in[mask_black] = (1.0 - omega) * P_in[mask_black] + omega * P_gs[mask_black]

            # convergence check
            residual = np.max(np.abs(P_new - P_old))
            if residual < tolerance:
                break

        # === Boundary conditions (2 ghost cell layers) ===
        # Left boundary (inlet) Neumann
        P_new[0, :] = P_new[2, :]
        P_new[1, :] = P_new[2, :]
        # Right boundary (outlet) reference pressure
        P_new[-1, :] = P_new[-3, :]
        P_new[-2, :] = P_new[-3, :]
        # Bottom and top walls Neumann
        P_new[:, 0] = P_new[:, 2]
        P_new[:, 1] = P_new[:, 2]
        P_new[:, -1] = P_new[:, -3]
        P_new[:, -2] = P_new[:, -3]

        self.P = P_new
    
    def correction_velocity(self, u_star, v_star):
        # Calculate pressure gradients (central differences) on interior [2:-2, 2:-2]
        i_slice = slice(2, -2)
        j_slice = slice(2, -2)
        
        dp_dx = (self.P[3:-1, j_slice] - self.P[1:-3, j_slice]) * self.inv_dx * 0.5
        dp_dy = (self.P[i_slice, 3:-1] - self.P[i_slice, 1:-3]) * self.inv_dy * 0.5

        # Apply correction: u^(n+1) = u* - (Δt/ρ)∇P
        u_new = np.copy(u_star)
        v_new = np.copy(v_star)
        
        u_new[i_slice, j_slice] = u_star[i_slice, j_slice] - (self.dt / const.rho) * dp_dx
        v_new[i_slice, j_slice] = v_star[i_slice, j_slice] - (self.dt / const.rho) * dp_dy

        # --- Left boundary (i=0,1) special handling for corrected velocity ---
        for i in [0, 1]:
            for j in range(2, self.m-2):
                v_loc = v_star[i, j]
                u_new[i, j] = 0  # No-slip
                
                # Upwind advection for v-component
                if v_loc >= 0:
                    adv_v_y = v_loc * (v_loc - v_star[i, j-1]) * self.inv_dy
                else:
                    adv_v_y = v_loc * (v_star[i, j+1] - v_loc) * self.inv_dy
                
                # Asymmetric diffusion
                diffusion_v = const.nu * (
                    (2*v_star[i+1, j] - 2*v_loc) * self.inv_dx**2 +
                    (v_star[i, j+1] - 2*v_loc + v_star[i, j-1]) * self.inv_dy**2
                )
                
                # Apply pressure correction (use gradient at i=2 for ghost cells)
                dp_dy_local = dp_dy[0, j-2] if i < 2 else dp_dy[i-2, j-2]
                v_new[i, j] = v_loc + self.dt * (-adv_v_y + diffusion_v - (self.dt / const.rho) * dp_dy_local)

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
                u_loc = u[i, j]
                v_loc = v[i, j]
                
                u_pos = max(u_loc, 0.0)
                u_neg = min(u_loc, 0.0)
                v_pos = max(v_loc, 0.0)
                v_neg = min(v_loc, 0.0)
                
                # Advection for u
                adv_x_u = (u_pos * (u_loc - u[i-1, j]) * inv_dx +
                          u_neg * (u[i+1, j] - u_loc) * inv_dx)
                adv_y_u = (v_pos * (u_loc - u[i, j-1]) * inv_dy +
                          v_neg * (u[i, j+1] - u_loc) * inv_dy)
                
                # Diffusion for u (4th order)
                diff_x_u = (-u[i+2, j] + 16.0*u[i+1, j] - 30.0*u_loc + 
                           16.0*u[i-1, j] - u[i-2, j]) * inv_dx**2 / 12.0
                diff_y_u = (-u[i, j+2] + 16.0*u[i, j+1] - 30.0*u_loc + 
                           16.0*u[i, j-1] - u[i, j-2]) * inv_dy**2 / 12.0
                diff_u = diffusivity * (diff_x_u + diff_y_u)
                
                u_new[i, j] = u_loc + dt * (-adv_x_u - adv_y_u + diff_u)
                
                # Advection for v
                adv_x_v = (u_pos * (v_loc - v[i-1, j]) * inv_dx +
                          u_neg * (v[i+1, j] - v_loc) * inv_dx)
                adv_y_v = (v_pos * (v_loc - v[i, j-1]) * inv_dy +
                          v_neg * (v[i, j+1] - v_loc) * inv_dy)
                
                # Diffusion for v (4th order)
                diff_x_v = (-v[i+2, j] + 16.0*v[i+1, j] - 30.0*v_loc + 
                           16.0*v[i-1, j] - v[i-2, j]) * inv_dx**2 / 12.0
                diff_y_v = (-v[i, j+2] + 16.0*v[i, j+1] - 30.0*v_loc + 
                           16.0*v[i, j-1] - v[i, j-2]) * inv_dy**2 / 12.0
                diff_v = diffusivity * (diff_x_v + diff_y_v)
                
                v_new[i, j] = v_loc + dt * (-adv_x_v - adv_y_v + diff_v)
        
        return u_new, v_new
    
    def adv_diff_interior_numba(self, inv_dx, inv_dy, src_u=None, src_v=None):
        """Numba-accelerated version of advection-diffusion computation."""
        return self._adv_diff_kernel(self.u, self.v, self.dt, inv_dx, inv_dy, 
                                     self.diffusivity, self.n, self.m)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _sor_kernel(P, u, v, inv_dx, inv_dy, rho, omega, tolerance, max_iterations, n, m):
        """Numba-optimized SOR solver kernel."""
        P_new = np.copy(P)
        
        # Compute RHS
        f = np.zeros_like(P)
        for i in prange(2, n-2):
            for j in range(2, m-2):
                du_dx = 0.5 * inv_dx * (u[i+1, j] - u[i-1, j])
                du_dy = 0.5 * inv_dy * (u[i, j+1] - u[i, j-1])
                dv_dx = 0.5 * inv_dx * (v[i+1, j] - v[i-1, j])
                dv_dy = 0.5 * inv_dy * (v[i, j+1] - v[i, j-1])
                
                f[i, j] = -rho * (du_dx**2 + 2.0 * du_dy * dv_dx + dv_dy**2)
        
        denom = 2.0 * (inv_dx**2 + inv_dy**2)
        
        # SOR iterations
        for iteration in range(max_iterations):
            P_old = P_new.copy()
            
            # Red points
            for i in prange(2, n-2):
                for j in range(2, m-2):
                    if (i + j) % 2 == 0:
                        laplacian = ((P_new[i+1, j] + P_new[i-1, j]) * inv_dx**2 +
                                    (P_new[i, j+1] + P_new[i, j-1]) * inv_dy**2)
                        P_gs = (laplacian - f[i, j]) / denom
                        P_new[i, j] = (1.0 - omega) * P_new[i, j] + omega * P_gs
            
            # Black points
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
        
        # Apply boundary conditions
        P_new[0, :] = P_new[2, :]
        P_new[1, :] = P_new[2, :]
        P_new[-1, :] = P_new[-3, :]
        P_new[-2, :] = P_new[-3, :]
        P_new[:, 0] = P_new[:, 2]
        P_new[:, 1] = P_new[:, 2]
        P_new[:, -1] = P_new[:, -3]
        P_new[:, -2] = P_new[:, -3]
        
        return P_new
    
    def SOR_pressure_solver_numba(self, u, v):
        """Numba-accelerated version of SOR pressure solver."""
        omega = 1.5
        tolerance = 1e-6
        max_iterations = 2000
        
        self.P = self._sor_kernel(self.P, u, v, self.inv_dx, self.inv_dy, 
                                  const.rho, omega, tolerance, max_iterations, 
                                  self.n, self.m)
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _correction_kernel(u_star, v_star, P, dt, inv_dx, inv_dy, rho, nu, n, m):
        """Numba-optimized velocity correction kernel."""
        u_new = np.copy(u_star)
        v_new = np.copy(v_star)
        
        # Interior correction
        for i in prange(2, n-2):
            for j in range(2, m-2):
                dp_dx = (P[i+1, j] - P[i-1, j]) * inv_dx * 0.5
                dp_dy = (P[i, j+1] - P[i, j-1]) * inv_dy * 0.5
                
                u_new[i, j] = u_star[i, j] - (dt / rho) * dp_dx
                v_new[i, j] = v_star[i, j] - (dt / rho) * dp_dy
        
        # Left boundary special handling
        for i in range(2):
            for j in range(2, m-2):
                v_loc = v_star[i, j]
                u_new[i, j] = 0.0
                
                if v_loc >= 0:
                    adv_v_y = v_loc * (v_loc - v_star[i, j-1]) * inv_dy
                else:
                    adv_v_y = v_loc * (v_star[i, j+1] - v_loc) * inv_dy
                
                diffusion_v = nu * (
                    (2*v_star[i+1, j] - 2*v_loc) * inv_dx**2 +
                    (v_star[i, j+1] - 2*v_loc + v_star[i, j-1]) * inv_dy**2
                )
                
                dp_dy_local = (P[2, j] - P[0, j]) * inv_dy * 0.5
                v_new[i, j] = v_loc + dt * (-adv_v_y + diffusion_v - (dt / rho) * dp_dy_local)
        
        return u_new, v_new
    
    def correction_velocity_numba(self, u_star, v_star):
        """Numba-accelerated version of velocity correction."""
        return self._correction_kernel(u_star, v_star, self.P, self.dt, 
                                      self.inv_dx, self.inv_dy, const.rho, 
                                      const.nu, self.n, self.m)