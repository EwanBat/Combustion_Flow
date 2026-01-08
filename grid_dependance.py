from chemistry import Chemistry, ChemistryManager
from fluid import Fluid
from system import System
import constant as const

import numpy as np
import matplotlib.pyplot as plt

def main():
    L_n = [25, 50, 75, 100, 150, 200]
    L_strainrate = []
    L_diffthick = []
    L_maxtemp = []

    
    dt_data = 1e-5
    total_time = 6e-3

    density = const.rho
    viscosity = const.nu  # Dynamic viscosity

    # Define chemistry properties for CH4 and O2
    chemistries = {
        'CH4': Chemistry(density=density, molar_mass=const.molar_mass['CH4'], diffusivity=viscosity),
        'O2': Chemistry(density=density, molar_mass=const.molar_mass['O2'], diffusivity=viscosity),
        'CO2': Chemistry(density=density, molar_mass=const.molar_mass['CO2'], diffusivity=viscosity),
        'H2O': Chemistry(density=density, molar_mass=const.molar_mass['H2O'], diffusivity=viscosity),
        'N2': Chemistry(density=density, molar_mass=const.molar_mass['N2'], diffusivity=viscosity)
    }
    parallel = True

    for n in L_n:
        m = n

        # Create Fluid object
        u_initial = np.zeros((n, m))  # m/s
        v_initial = np.zeros((n, m))  # m/s
        P_initial = np.zeros((n, m))  # Pa
        fluid = Fluid(n=n, m=m, diffusivity=viscosity, rho=density)
        fluid.velocity_initialization(u_initial, v_initial, P_initial)

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

        system = System(dt_data=dt_data, total_time=total_time, n=n, m=m, fluid=fluid, ChemicalManager=chemistry_manager)
        system.print_caracteristics()
        system.run(parallel=parallel)
        system.flow_field_info()

        L_strainrate.append(system.strain_rate_left)
        L_diffthick.append(system.diffusive_thickness)
        L_maxtemp.append(np.max(system.T))
    
    # Plotting results
    plt.figure(figsize=(8, 6))
    plt.plot(L_n, L_strainrate, marker='o')
    plt.xlabel('Grid Size (n)')
    plt.ylabel('Strain Rate on Left Wall (1/s)')
    plt.title('Strain Rate vs Grid Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('strain_rate_vs_grid_size.png')

    plt.figure(figsize=(8, 6))
    plt.plot(L_n, L_diffthick, marker='o')
    plt.xlabel('Grid Size (n)')
    plt.ylabel('Diffusive Zone Thickness on Left Wall (m)')
    plt.title('Diffusive Zone Thickness vs Grid Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('diffusive_thickness_vs_grid_size.png')

    plt.figure(figsize=(8, 6))
    plt.plot(L_n, L_maxtemp, marker='o')
    plt.xlabel('Grid Size (n)')
    plt.ylabel('Maximum Temperature (K)')
    plt.yscale('log')
    plt.title('Maximum Temperature vs Grid Size')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('max_temperature_vs_grid_size.png')

if __name__ == "__main__":
    main()