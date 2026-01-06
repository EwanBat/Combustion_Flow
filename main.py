from chemistry import Chemistry, ChemistryManager
from fluid import Fluid
from system import System
import constant as const

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define fluid properties
    density = const.rho
    viscosity = const.nu  # Dynamic viscosity

    # Initialize velocity field (example: uniform flow in x-direction)
    n, m = 200, 200  # Grid size
    
    # Create Fluid object
    u_initial = np.zeros((n, m))  # m/s
    v_initial = np.zeros((n, m))  # m/s
    P_initial = np.zeros((n, m))  # Pa
    fluid = Fluid(n=n, m=m, diffusivity=viscosity)
    fluid.velocity_initialization(u_initial, v_initial, P_initial)

    # Define chemistry properties for CH4 and O2
    chemistries = {
        'CH4': Chemistry(density=density, molar_mass=const.molar_mass['CH4'], diffusivity=viscosity),
        'O2': Chemistry(density=density, molar_mass=const.molar_mass['O2'], diffusivity=viscosity),
        'CO2': Chemistry(density=density, molar_mass=const.molar_mass['CO2'], diffusivity=viscosity),
        'H2O': Chemistry(density=density, molar_mass=const.molar_mass['H2O'], diffusivity=viscosity),
        'N2': Chemistry(density=density, molar_mass=const.molar_mass['N2'], diffusivity=viscosity)
    }

    # Initialize mass fraction fields (example: CH4 in left half, O2 in right half)
    Y_CH4_initial = np.zeros((n, m))
    Y_CO2_initial = np.zeros((n, m))
    Y_H2O_initial = np.zeros((n, m))
    Y_N2_initial = np.zeros((n, m)) # Assuming air is 79% N2
    Y_O2_initial = np.zeros((n, m)) # Assuming air is 21% O2

    # Initialize the system
    chemistries['CH4'].Y_initialization(Y_CH4_initial)
    chemistries['O2'].Y_initialization(Y_O2_initial)
    chemistries['CO2'].Y_initialization(Y_CO2_initial)
    chemistries['H2O'].Y_initialization(Y_H2O_initial)
    chemistries['N2'].Y_initialization(Y_N2_initial)

    chemistry_manager = ChemistryManager(n=n, m=m, chem_dict=chemistries)

    dt_data = 1e-5
    total_time = 6e-3
    system = System(dt_data=dt_data, total_time=total_time, n=n, m=m, fluid=fluid, ChemicalManager=chemistry_manager)
    system.print_caracteristics()
    # parallel = input("Run simulation with Numba parallelization? (y/n): ").strip().lower() == 'y'
    # system.run(parallel=parallel)
    system.plot_concentration(save = True)
    system.plot_temperature(save = True)
    system.plot_velocity_magnitude(save = True)
    system.plot_divergence(save = True)
    system.animation_concentration(species="CO2", save=True)
    system.animation_temperature(save=True)
    # system.flow_field_info()

if __name__ == "__main__":
    main()