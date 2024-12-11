# Nanoparticle Orbital Simulator

A Flask web application for simulating the orbital motion of charged nanoparticles in electromagnetic fields. The simulation uses the Lorentz force equations to calculate the particle's trajectory in 3D space.

## Features

- Interactive 3D visualization of particle trajectories
- Adjustable parameters:
  - Particle mass and charge
  - Initial position and velocity
  - Magnetic field strength and direction
- Real-time simulation updates
- Scientific computation using NumPy and SciPy
- Modern web interface with Plotly.js

## Installation

You can install the dependencies using either conda or pip.

### Using Conda (Recommended)

1. Clone this repository:
```bash
git clone https://github.com/yourusername/orbital-simulations.git
cd orbital-simulations
```

2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate orbital-sim
```

### Using Pip

1. Clone this repository:
```bash
git clone https://github.com/yourusername/orbital-simulations.git
cd orbital-simulations
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://localhost:5000`

3. Adjust the simulation parameters:
   - Set the particle's mass and charge
   - Define initial position and velocity
   - Configure the magnetic field components
   
4. Click "Run Simulation" to visualize the particle's trajectory

## Physics Background

The simulation uses the Lorentz force equation:
F = q(E + v × B)

where:
- F is the force on the charged particle
- q is the particle's charge
- E is the electric field
- v is the particle's velocity
- B is the magnetic field

## License

MIT License 