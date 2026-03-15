import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from matplotlib import gridspec

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
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.5)
    
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
    
def plot_divergence(self, load_from="data//simulation_data", save=False, filename="divergence.png"):
    """
    Plot divergence of the velocity field.
    
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
    
    # Calculate divergence
    dudx = (u_plot[2:, 1:-1] - u_plot[:-2, 1:-1]) / (2 * self.dx)
    dvdy = (v_plot[1:-1, 2:] - v_plot[1:-1, :-2]) / (2 * self.dy)
    divergence = dudx + dvdy
    
    # Create figure
    fig = plt.figure(figsize=(8, 6))
    
    # Plot divergence
    plt.imshow(divergence.T, extent=(self.dx*1e3, (self.Lx - self.dx)*1e3,
                                        self.dy*1e3, (self.Ly - self.dy)*1e3), 
                aspect='auto', cmap='seismic', vmin=-np.max(np.abs(divergence[5:-5,5-5])), vmax=np.max(np.abs(divergence[5:-5,5-5])))
    
    plt.colorbar(label='Divergence (1/s)')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    plt.title('Velocity Field Divergence')
    plt.tight_layout()
    
    if save:
        plt.savefig(filename)
    
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

def flow_field_info(self):
    # Calculate the strain rate on the left wall
    v = np.load("data//simulation_data_v.npy")
    dv_dy_left = (v[0, 2:-2] - v[0, 0:-4]) / (2 * self.dy)
    strain_rate_left = np.max(np.abs(dv_dy_left))

    # Measure the thickness along x of the diffusive zone on the left wall using the N2 species between Y=0.1 N2max and Y=0.9 N2max
    Y_N2 = np.load("data//simulation_data_Y_N2.npy")
    Y_N2_max = np.max(Y_N2)
    x_coords = np.linspace(0.0, self.Lx, self.n)
    for j in range(self.m):
        ind_10 = []
        ind_90 = []
        Y_profile = Y_N2[:,j]
        for i in range(self.n):
            if Y_profile[i] >= 0.1 * Y_N2_max:
                ind_10.append(i)
            if Y_profile[i] <= 0.9 * Y_N2_max:
                ind_90.append(i)
        if ind_10 and ind_90:
            x_10 = x_coords[ind_10[0]]
            x_90 = x_coords[ind_90[-1]]
            thickness = x_90 - x_10
            if j == 0:
                diffusive_thickness = thickness
            else:
                diffusive_thickness = min(diffusive_thickness, thickness)  
    
    print("=" * 50)
    print("FLOW FIELD INFORMATION")
    print("strain rate on left wall: {:.4f} 1/s".format(strain_rate_left))
    if diffusive_thickness is not None:
        print("diffusive zone thickness on left wall: {:.4e} m".format(diffusive_thickness))
    
    return strain_rate_left, diffusive_thickness
