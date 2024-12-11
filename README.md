# Orbital Simulations

A flask deployment of multiple scientific python libraries for material verification and visualisations.

Built to complement a literature-based scientific discovery pipeline.

Built as part of a material science pipeline, using NanoBERT and NanoSearch for the extrraction of relevant information from the literature. To this effect, Molecular structure analysis and MD simulation software is used.

materials science visualisations.

A comprehensive web application for simulating and visualizing various quantum, classical, and materials science phenomena. 

Built with Flask and modern scientific Python libraries.

## Core Modules

### 1. Classical Physics (`classical_trajectory.py`)
**Short**: Simulates charged particle trajectories in electromagnetic fields.
**Detailed**: 
- Implements relativistic equations of motion for charged particles
- Handles arbitrary E and B field configurations
- Uses RK4 integration for high accuracy
- Visualizes 3D trajectories with Plotly
- Includes effects like:
  - Lorentz force
  - Relativistic corrections
  - Time dilation
  - Field interactions

### 2. Quantum Orbitals (`quantum_orbitals.py`)
**Short**: Calculates and visualizes atomic/molecular orbitals.
**Detailed**:
- Computes hydrogen-like wavefunctions using:
  - Spherical harmonics for angular part
  - Laguerre polynomials for radial part
- Supports multiple atoms (H to Ne)
- Calculates electron density and probability distributions
- Features:
  - Energy level calculations
  - Orbital visualization (s, p, d orbitals)
  - Electron density plots
  - Quantum number selection

### 3. Semi-Classical (`semiclassical.py`)
**Short**: Implements semi-classical approximations for quantum systems.
**Detailed**:
- WKB approximation for tunneling problems
- Bohr-Sommerfeld quantization
- Phase space trajectories
- Features:
  - Tunneling probability calculations
  - Barrier penetration
  - Connection formulas
  - Classical turning points

### 4. Double Slit (`double_slit.py`)
**Short**: Simulates quantum double-slit experiment.
**Detailed**:
- Time-dependent Schrödinger equation solver
- Quantum wave packet evolution
- Interference pattern calculation
- Features:
  - Adjustable slit parameters
  - Wave packet properties
  - Detection screen simulation
  - Probability distribution

### 5. Particle-In-Cell (`pic_simulation.py`)
**Short**: PIC simulation for plasma physics.
**Detailed**:
- Full electromagnetic PIC implementation
- Features:
  - Charge deposition (Cloud-in-Cell)
  - Field solver (FFT-based Poisson)
  - Particle pusher (Boris algorithm)
  - Diagnostics:
    - Energy conservation
    - Phase space
    - Field evolution
    - Particle distributions

### 6. Shape Optimization (`shape_optimization.py`)
**Short**: Optimizes nanostructure shapes for specific properties.
**Detailed**:
- Gradient-based optimization
- Supports multiple objectives:
  - Optical response
  - Surface area
  - Volume constraints
- Features:
  - Parametric shape representation
  - Automatic differentiation
  - Multi-objective optimization
  - Constraint handling

### 7. Battery Simulation (`battery_simulation.py`)
**Short**: Simulates Li-ion battery behavior.
**Detailed**:
- Models S/PCNF/CNT composite material
- Features:
  - Capacity fade modeling
  - Voltage profiles
  - Temperature effects
  - Cycling behavior
  - Performance metrics:
    - Capacity retention
    - Voltage efficiency
    - Energy density
    - Cycle life

### 8. QuTiP Dynamics (`qutip_simulation.py`)
**Short**: Quantum system dynamics using QuTiP.
**Detailed**:
- Quantum master equation solver
- Features:
  - State evolution
  - Decoherence effects
  - Quantum gates
  - Visualization:
    - Bloch sphere
    - Density matrices
    - Expectation values
    - Wigner functions

### 9. ASE Simulation (`ase_simulation.py`)
**Short**: Atomistic simulations using ASE.
**Detailed**:
- Atomic structure manipulation
- Features:
  - Structure creation (bulk, surface, nanoparticle)
  - Geometry optimization
  - Molecular dynamics
  - Analysis:
    - Energy calculation
    - Structure visualization
    - Trajectory analysis
    - Temperature control

### 10. PyVista Visualization (`pyvista_visualization.py`)
**Short**: 3D visualization of nanostructures.
**Detailed**:
- Nanoparticle visualization types:
  - Spherical
  - Rod-shaped
  - Cubic
  - Core-shell
- Features:
  - Surface ligands
  - Electrostatic potentials
  - Custom colormaps
  - Interactive rotation/zoom

### 11. Pymatgen Analysis (`pymatgen_analysis.py`)
**Short**: Materials analysis using pymatgen.
**Detailed**:
- Crystal structure analysis
- Features:
  - Structure creation
  - Symmetry analysis
  - Defect creation
  - Surface generation
  - Analysis:
    - Bonding environment
    - Electronic structure
    - Surface properties
    - Strain effects

### 12. Quantum ESPRESSO (`qe_simulation.py`)
**Short**: DFT calculations using Quantum ESPRESSO.
**Detailed**:
- First-principles calculations
- Features:
  - Structure optimization
  - Band structure
  - Density of states
  - Electronic properties
  - Analysis:
    - K-point sampling
    - Convergence testing
    - Pseudopotential selection
    - Post-processing

### 13. NanoHUB Tools (`nanohub_simulation.py`)
**Short**: Interface with NanoHUB simulation tools.
**Detailed**:
- Quantum dot simulations:
  - Energy levels
  - Material selection
  - Temperature effects
  - Quantum numbers
- Carbon nanotube simulations:
  - Band structure
  - Chiral vectors
  - Length effects
  - K-point sampling

## Tools and Libraries Used

### Visualization
- **Plotly**: Interactive plots for all simulations
- **PyVista**: 3D visualization of nanostructures
- **Matplotlib**: Static plots and analysis

### Scientific Computing
- **NumPy**: Core numerical computations
- **SciPy**: Scientific algorithms and optimization
- **SymPy**: Symbolic mathematics
- **pandas**: Data analysis and manipulation

### Quantum Chemistry
- **QuTiP**: Quantum dynamics
- **ASE**: Atomic simulations
- **Pymatgen**: Materials analysis
- **Quantum ESPRESSO**: DFT calculations

### Web Framework
- **Flask**: Web application backend
- **HTML/CSS/JavaScript**: Frontend interface
- **AJAX**: Asynchronous updates

### Additional Tools
- **hublib**: NanoHUB integration
- **spglib**: Symmetry analysis
- **seekpath**: K-path generation
- **PIL**: Image processing

## Installation

1. Create conda environment:
```bash
conda env create -f environment.yml
```

2. Activate environment:
```bash
conda activate orbitals
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open web browser and navigate to:
```
http://localhost:5000
```

3. Select simulation type from the tabs
4. Adjust parameters as needed
5. Run simulation and view results

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request 