# Numerical Methods for Reactive Flow Simulation

A 2D computational fluid dynamics (CFD) solver for simulating reactive flows with coupled fluid dynamics and chemical kinetics.

## Overview

This project implements a numerical simulation of a counterflow premixed flame using:
- **Incompressible Navier-Stokes solver** with fractional step method
- **Multi-species reactive chemistry** with Arrhenius kinetics
- **Operator splitting** for advection-diffusion-reaction equations
- **Runge-Kutta 4th order (RK4)** time integration for species and temperature

## Features

- 2D incompressible flow solver with upwind advection scheme
- Multi-species chemistry manager (CH₄, O₂, CO₂, H₂O, N₂)
- Temperature-dependent reaction rates (Arrhenius law)
- Pressure correction using Successive Over-Relaxation (SOR)
- Automatic time step selection based on CFL and Fourier criteria
- Data export and visualization tools
- Animation generation for temporal evolution

## Project Structure

```
nm_method/
├── main.py           # Entry point and simulation setup
├── system.py         # Main system controller
├── fluid.py          # Fluid dynamics solver
├── chemistry.py      # Chemistry manager and species handling
├── constant.py       # Physical constants and parameters
├── data/             # Simulation output data (*.npy files)
├── .gitignore        # Git ignore rules
└── README.md         # This file
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy (optional, for advanced features)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/EwanBat/combustion_flow.git
cd combustion_flow
```

2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Usage

Run the simulation:
```bash
python main.py
```

The simulation will:
1. Initialize the computational domain and fields
2. Run the time-stepping loop with automatic progress display
3. Save data snapshots at specified intervals to `data/` folder
4. Generate visualization plots and animations

### Customization

Edit parameters in `constant.py` to modify:
- Physical properties (density, viscosity, specific heat)
- Reaction kinetics parameters (activation energy, pre-exponential factor)
- Species properties (molecular mass, diffusivity)

Edit `main.py` to adjust:
- Domain size and grid resolution
- Inlet velocities and boundary conditions
- Total simulation time and data save frequency

## Physics

### Governing Equations

**Momentum (Navier-Stokes):**

$$ \frac{\partial \vec{u}}{\partial t} + \vec{u} \cdot \nabla \vec{u} = -\frac{\nabla P}{\rho} + \nu \nabla^2 \vec{u} $$
$$ \nabla \cdot \vec{u} = 0 \quad \text{(incompressibility)} $$

**Species transport:**

$$ \frac{\partial Y_i}{\partial t} + \vec{u} \cdot \nabla Y_i = D \nabla^2 Y_i + \frac{\omega_i}{\rho} $$

**Energy (temperature):**

$$ \frac{\partial T}{\partial t} + \vec{u} \cdot \nabla T = \alpha \nabla^2 T + \frac{Q}{\rho c_p} $$

### Reaction Model

Simple one-step methane combustion:

$$ \text{CH}_4 + 2\text{O}_2 \rightarrow \text{CO}_2 + 2\text{H}_2\text{O} $$

With Arrhenius rate:

$$ \omega = A \cdot \exp\left(-\frac{E_a}{RT}\right) \cdot [\text{CH}_4] \cdot [\text{O}_2]^2 $$

## Visualization

The code generates:
- **Velocity magnitude plots** showing flow patterns
- **Temperature fields** with heated regions
- **Species concentration plots** for each chemical species
- **Animations** showing temporal evolution

## Output

Simulation data is saved as NumPy arrays (`.npy`) in the `data/` folder:
- `simulation_data_t0000_u.npy` - x-velocity at timestep 0
- `simulation_data_t0000_v.npy` - y-velocity at timestep 0
- `simulation_data_t0000_T.npy` - temperature at timestep 0
- `simulation_data_t0000_Y_CH4.npy` - CH₄ mass fraction at timestep 0
- ... (and so on for each timestep and species)

## License

This project is provided for educational purposes.

## Authors

- Ewan Bataille

## Acknowledgments

Developed as part of a numerical methods course project.
