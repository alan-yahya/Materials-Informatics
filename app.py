from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.integrate import odeint
import plotly.express as px
import plotly.graph_objects as go
import json
from quantum_orbitals import (
    create_atom, calculate_orbital_wavefunction, plot_orbital_density,
    get_available_atoms, get_available_basis_sets
)
from double_slit import run_double_slit_simulation
from battery_simulation import run_battery_simulation

app = Flask(__name__)

def orbital_equations(state, t, mass, charge, electric_field, magnetic_field):
    """
    Equations of motion for charged nanoparticle in electromagnetic fields
    state = [x, y, z, vx, vy, vz]
    """
    x, y, z, vx, vy, vz = state
    
    # Lorentz force equations
    ax = (charge/mass) * (electric_field[0] + vy*magnetic_field[2] - vz*magnetic_field[1])
    ay = (charge/mass) * (electric_field[1] + vz*magnetic_field[0] - vx*magnetic_field[2])
    az = (charge/mass) * (electric_field[2] + vx*magnetic_field[1] - vy*magnetic_field[0])
    
    return [vx, vy, vz, ax, ay, az]

@app.route('/')
def index():
    atoms = get_available_atoms()
    basis_sets = get_available_basis_sets()
    return render_template('index.html', atoms=atoms, basis_sets=basis_sets)

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()
    
    # Get simulation parameters from request
    mass = float(data.get('mass', 1e-20))  # kg
    charge = float(data.get('charge', 1.6e-19))  # Coulombs
    
    # Initial conditions
    initial_position = data.get('initial_position', [0, 0, 0])
    initial_velocity = data.get('initial_velocity', [100, 0, 0])
    
    # Fields
    electric_field = data.get('electric_field', [0, 0, 0])
    magnetic_field = data.get('magnetic_field', [0, 0, 1])
    
    # Time array
    t = np.linspace(0, 1e-6, 1000)
    
    # Initial state
    initial_state = initial_position + initial_velocity
    
    # Solve ODE
    solution = odeint(orbital_equations, initial_state, t, 
                     args=(mass, charge, electric_field, magnetic_field))
    
    # Create 3D trajectory plot
    fig = go.Figure(data=[go.Scatter3d(
        x=solution[:, 0],
        y=solution[:, 1],
        z=solution[:, 2],
        mode='lines',
        line=dict(color='blue', width=2)
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)'
        ),
        title='Nanoparticle Orbital Trajectory'
    )
    
    return jsonify({
        'plot': fig.to_json(),
        'trajectory': solution.tolist()
    })

@app.route('/quantum_orbital', methods=['POST'])
def quantum_orbital():
    data = request.get_json()
    
    # Get quantum parameters
    atom_symbol = data.get('atom', 'H')
    basis = data.get('basis', '6-31g')
    grid_points = int(data.get('grid_points', 50))
    radius = float(data.get('radius', 5.0))
    isovalue = float(data.get('isovalue', 0.01))
    
    # Create atom and calculate wavefunction
    atom = create_atom(atom_symbol, basis)
    x, y, z, density = calculate_orbital_wavefunction(atom, grid_points, radius)
    
    # Create visualization
    fig = plot_orbital_density(x, y, z, density, isovalue)
    
    return jsonify({
        'plot': fig.to_json()
    })

@app.route('/double_slit', methods=['POST'])
def double_slit():
    data = request.get_json()
    
    # Get simulation parameters
    slit_width = float(data.get('slit_width', 10))
    slit_separation = float(data.get('slit_separation', 30))
    wavelength = float(data.get('wavelength', 500e-9))
    
    # Run simulation
    fig = run_double_slit_simulation(
        slit_width=slit_width,
        slit_separation=slit_separation,
        wavelength=wavelength
    )
    
    return jsonify({
        'plot': fig.to_json()
    })

@app.route('/battery', methods=['POST'])
def battery():
    data = request.get_json()
    
    # Get simulation parameters
    initial_capacity = float(data.get('initial_capacity', 900))  # mAh/g
    cycles = int(data.get('cycles', 100))
    
    # Run simulation
    fig = run_battery_simulation(
        initial_capacity=initial_capacity,
        cycles=cycles
    )
    
    return jsonify({
        'plot': fig.to_json()
    })

if __name__ == '__main__':
    app.run(debug=True) 