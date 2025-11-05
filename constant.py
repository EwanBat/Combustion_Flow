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

c_p = 1200 # J/(kg K)
rho = 1.1614 # kg/m^3
nu = 15e-6 # m^2/s
R = 8.314 # J/(mol K)