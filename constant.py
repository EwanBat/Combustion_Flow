Lx = 2e-2              # Domain length in x (m)
Ly = 2e-2              # Domain width in y (m)
Lslot = 0.5e-2         # Length of fuel/oxidizer slot (m)
Lcoflow = 0.5e-2       # Length of coflow region (m)

Uslot = 10.0            # Slot inlet velocity (m/s)
Tslot = 300.0          # Slot inlet temperature (K)
Ucoflow = 2          # Coflow inlet velocity (m/s)
Tcoflow = 300.0        # Coflow inlet temperature (K)

T_rode = 1000       # Rod temperature (K)
width_rode = 2e-3       # Rod width (m)

c_p = 1200 # J/(kg K)
rho = 1.1614 # kg/m^3
alpha = 15e-6 # m^2/s
D = 15e-6 # m^2/s
nu = 15e-6 # m^2/s
R = 8.314 # J/(mol K)

A = 1.1e8
T_A = 1e4 # K

stoichiometric_coefficients = {
    'CH4': -1,
    'O2': -2,
    'CO2': 1,
    'H2O': 2,
    'N2': 0
}

enthalpy_of_formation = {
    'CH4': -74900, # J/mol
    'O2': 0,       # J/mol
    'CO2': -393520, # J/mol
    'H2O': -241818, # J/mol
    'N2': 0        # J/mol
}

molar_mass = {
    'CH4': 16.04e-3, # kg/mol
    'O2': 32.00e-3,  # kg/mol
    'CO2': 44.01e-3, # kg/mol
    'H2O': 18.02e-3, # kg/mol
    'N2': 28.01e-3   # kg/mol
}