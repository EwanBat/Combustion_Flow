import numpy as np
import constant as const

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
        
        i_slice = slice(1, -1); j_slice = slice(1, -1)
        u_loc = self.u[i_slice, j_slice]; v_loc = self.v[i_slice, j_slice]
        u_pos = np.maximum(u_loc, 0); u_neg = np.minimum(u_loc, 0)
        v_pos = np.maximum(v_loc, 0); v_neg = np.minimum(v_loc, 0)
        
        # Update u component
        adv_x_u = (u_pos * (u_loc - self.u[:-2, j_slice]) * inv_dx +
                    u_neg * (self.u[2:, j_slice] - u_loc) * inv_dx)
        adv_y_u = (v_pos * (u_loc - self.u[i_slice, :-2]) * inv_dy +
                    v_neg * (self.u[i_slice, 2:] - u_loc) * inv_dy)
        diff_x_u = (self.u[2:, j_slice] - 2*u_loc + self.u[:-2, j_slice]) * inv_dx**2
        diff_y_u = (self.u[i_slice, 2:] - 2*u_loc + self.u[i_slice, :-2]) * inv_dy**2
        diff_u = self.diffusivity * (diff_x_u + diff_y_u)
        # src_term_u = src_u[i_slice, j_slice] if src_u is not None else 0.0
        u_new[i_slice, j_slice] = u_loc + self.dt * (-adv_x_u - adv_y_u + diff_u )

        # Update v component
        adv_x_v = (u_pos * (v_loc - self.v[:-2, j_slice]) * inv_dx +
                    u_neg * (self.v[2:, j_slice] - v_loc) * inv_dx)
        adv_y_v = (v_pos * (v_loc - self.v[i_slice, :-2]) * inv_dy +
                    v_neg * (self.v[i_slice, 2:] - v_loc) * inv_dy)
        diff_x_v = (self.v[2:, j_slice] - 2*v_loc + self.v[:-2, j_slice]) * inv_dx**2
        diff_y_v = (self.v[i_slice, 2:] - 2*v_loc + self.v[i_slice, :-2]) * inv_dy**2
        diff_v = self.diffusivity * (diff_x_v + diff_y_v)
        # src_term_v = src_v[i_slice, j_slice] if src_v is not None else 0.0
        v_new[i_slice, j_slice] = v_loc + self.dt * (-adv_x_v - adv_y_v + diff_v)

        return u_new, v_new

    # Optionnel : méthode pour appliquer BCs de vitesse si on veut externaliser
    def apply_velocity_bcs(self, u_upd, v_upd, ind_inlet, ind_coflow, Uslot, Ucoflow):        
        # CH4 inlet (slot region, bottom wall)
        u_upd[:ind_inlet, 0] = 0
        v_upd[:ind_inlet, 0] = Uslot
        
        # O2+N2 inlet (slot region, top wall)
        u_upd[:ind_inlet, self.m-1] = 0
        v_upd[:ind_inlet, self.m-1] = -Uslot
        
        # N2 coflow inlet (coflow region, bottom wall)
        u_upd[ind_inlet:ind_coflow, 0] = 0
        v_upd[ind_inlet:ind_coflow, 0] = Ucoflow
        
        # N2 coflow inlet (coflow region, top wall)
        u_upd[ind_inlet:ind_coflow, self.m-1] = 0
        v_upd[ind_inlet:ind_coflow, self.m-1] = -Ucoflow
        
        # Lower wall (outlet region)
        u_upd[ind_coflow:, 0] = 0
        v_upd[ind_coflow:, 0] = 0
        
        # Upper wall (outlet region)
        u_upd[ind_coflow:, self.m-1] = 0
        v_upd[ind_coflow:, self.m-1] = 0
        
        # Right boundary (outlet - extrapolation)
        u_upd[self.n-1, 1:self.m-1] = u_upd[self.n-2, 1:self.m-1]
        v_upd[self.n-1, 1:self.m-1] = v_upd[self.n-2, 1:self.m-1]
        
    def SOR_pressure_solver(self, u, v):
        """
        Solve pressure Poisson equation using SOR for:
        ∇²p = -ρ [ (∂u/∂x)^2 + 2 (∂u/∂y)(∂v/∂x) + (∂v/∂y)^2 ]
        Uses central differences for derivatives and red-black SOR.
        Args:
            inv_dx: 1/dx
            inv_dy: 1/dy
            u, v: velocity fields (same grid as self.P)
        """
        # --- Compute derivatives with central differences on interior ---
        # ∂u/∂x and ∂v/∂y already used shape (n-2, m-2)
        du_dx = 0.5 * self.inv_dx * (u[2:, 1:-1] - u[:-2, 1:-1])
        du_dy = 0.5 * self.inv_dy * (u[1:-1, 2:] - u[1:-1, :-2])
        dv_dx = 0.5 * self.inv_dx * (v[2:, 1:-1] - v[:-2, 1:-1])
        dv_dy = 0.5 * self.inv_dy * (v[1:-1, 2:] - v[1:-1, :-2])

        # RHS: -rho * [ (du_dx)^2 + 2*(du_dy)*(dv_dx) + (dv_dy)^2 ]
        f = np.zeros_like(self.P)
        f_interior = -const.rho * (du_dx**2 + 2.0 * du_dy * dv_dx + dv_dy**2)
        f[1:-1, 1:-1] = f_interior

        # === Initialize SOR ===
        P_new = np.copy(self.P)
        omega = 1.5
        tolerance = 1e-6
        max_iterations = 2000

        interior = (slice(1, -1), slice(1, -1))
        f_in = f[1:-1, 1:-1]
        denom = 2.0 * (self.inv_dx**2 + self.inv_dy**2)

        # Precompute index masks for red-black pattern
        ii, jj = np.indices((self.n-2, self.m-2))
        mask_red = ((ii + jj) % 2) == 0
        mask_black = ~mask_red

        def compute_Pgs_local(P):
            P_ip = P[2:, 1:-1]      # P(i+1, j)
            P_im = P[:-2, 1:-1]     # P(i-1, j)
            P_jp = P[1:-1, 2:]      # P(i, j+1)
            P_jm = P[1:-1, :-2]     # P(i, j-1)
            laplacian = (P_ip + P_im) * self.inv_dx**2 + (P_jp + P_jm) * self.inv_dy**2
            return (laplacian - f_in) / denom

        for iteration in range(max_iterations):
            P_old = P_new.copy()

            P_in = P_new[1:-1, 1:-1]

            # Update red points
            P_gs = compute_Pgs_local(P_new)
            P_in[mask_red] = (1.0 - omega) * P_in[mask_red] + omega * P_gs[mask_red]

            # Update black points (recompute Gauss-Seidel right-hand side)
            P_gs = compute_Pgs_local(P_new)
            P_in[mask_black] = (1.0 - omega) * P_in[mask_black] + omega * P_gs[mask_black]

            # convergence check (max change)
            residual = np.max(np.abs(P_new - P_old))
            if residual < tolerance:
                break

        # === Boundary conditions ===
        # Left boundary (inlet) Neumann
        P_new[0, :] = P_new[1, :]
        # Right boundary (outlet) reference pressure (Dirichlet)
        P_new[-1, :] = 0.0
        # Bottom and top walls Neumann
        P_new[:, 0] = P_new[:, 1]
        P_new[:, -1] = P_new[:, -2]

        self.P = P_new
    
    def correction_velocity(self, u_star, v_star):
        # Calculate pressure gradients (central differences)
        dp_dx = (self.P[2:, 1:-1] - self.P[:-2, 1:-1]) * self.inv_dx * 0.5
        dp_dy = (self.P[1:-1, 2:] - self.P[1:-1, :-2]) * self.inv_dy * 0.5

        # Apply correction: u^(n+1) = u* - (Δt/ρ)∇P
        u_new = np.copy(u_star)
        v_new = np.copy(v_star)
        i_int = slice(1, -1); j_int = slice(1, -1)
        u_new[i_int, j_int] = u_star[i_int, j_int] - (self.dt / const.rho) * dp_dx
        v_new[i_int, j_int] = v_star[i_int, j_int] - (self.dt / const.rho) * dp_dy

        # --- Left boundary (i=0) special handling for corrected velocity ---
        i = 0
        for j in range(1, self.m-1):
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
            
            # Apply pressure correction
            v_new[i, j] = v_loc + self.dt * (-adv_v_y + diffusion_v - (self.dt / const.rho) * dp_dy[i, j-1])

        return u_new, v_new