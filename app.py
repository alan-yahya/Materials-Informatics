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
from semiclassical import run_semiclassical_simulation
from pic_simulation import run_pic_simulation
from shape_optimization import run_shape_optimization
from qutip_simulation import run_qutip_simulation
from ase_simulation import run_ase_simulation
from pyvista_visualization import run_pyvista_visualization
from pymatgen_analysis import run_pymatgen_analysis
from qe_simulation import run_qe_simulation
from chemml_analysis import run_chemml_analysis

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

@app.route('/semiclassical', methods=['POST'])
def semiclassical():
    data = request.get_json()
    
    # Get simulation parameters
    mass = float(data.get('mass', 9.1093837015e-31))  # electron mass in kg
    energy = float(data.get('energy', 1.0))  # energy in eV
    potential_type = data.get('potential_type', 'barrier')
    
    # Run simulation
    fig = run_semiclassical_simulation(
        mass=mass,
        energy=energy,
        potential_type=potential_type
    )
    
    return jsonify({
        'plot': fig.to_json()
    })

@app.route('/pic', methods=['POST'])
def pic():
    data = request.get_json()
    
    # Get simulation parameters
    n_particles = int(data.get('n_particles', 1000))
    n_steps = int(data.get('n_steps', 100))
    nx = int(data.get('nx', 100))
    ny = int(data.get('ny', 100))
    dt = float(data.get('dt', 1e-12))
    
    # Run simulation
    fig = run_pic_simulation(
        n_particles=n_particles,
        n_steps=n_steps,
        nx=nx,
        ny=ny,
        dt=dt
    )
    
    return jsonify({
        'plot': fig.to_json()
    })

@app.route('/optimize_shape', methods=['POST'])
def optimize_shape():
    data = request.get_json()
    
    # Get optimization parameters
    n_points = int(data.get('resolution', 50))  # Using resolution as n_points
    max_iter = int(data.get('max_iter', 50))
    
    # Run optimization
    fig = run_shape_optimization(
        n_points=n_points,
        max_iter=max_iter
    )
    
    return jsonify({
        'plot': fig.to_json()
    })

@app.route('/qutip_template')
def qutip_template():
    return render_template('qutip.html')

@app.route('/qutip', methods=['POST'])
def qutip():
    try:
        data = request.get_json()
        print("Received QuTiP request with data:", data)  # Debug log
        
        # Get simulation parameters
        n_levels = int(data.get('n_levels', 3))
        gamma = float(data.get('gamma', 0.1))
        omega = float(data.get('omega', 1.0))
        t_max = float(data.get('t_max', 10.0))
        n_steps = int(data.get('n_steps', 200))
        
        print(f"Parameters: n_levels={n_levels}, gamma={gamma}, omega={omega}, t_max={t_max}, n_steps={n_steps}")  # Debug log
        
        # Run simulation
        fig = run_qutip_simulation(
            n_levels=n_levels,
            gamma=gamma,
            omega=omega,
            t_max=t_max,
            n_steps=n_steps
        )
        
        print("Simulation completed successfully")  # Debug log
        response = jsonify({'plot': fig.to_json()})
        print("Response created")  # Debug log
        return response
        
    except Exception as e:
        print(f"Error in QuTiP route: {str(e)}")  # Debug log
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/ase', methods=['POST'])
def ase():
    try:
        data = request.get_json()
        print("Received ASE request with data:", data)
        
        # Get simulation parameters
        structure_type = data.get('structure_type', 'bulk')
        material = data.get('material', 'Cu')
        size = tuple(data.get('size', [2, 2, 2]))
        vacuum = float(data.get('vacuum', 10.0))
        temperature = float(data.get('temperature', 300))
        timestep = float(data.get('timestep', 1.0))
        steps = int(data.get('steps', 100))
        
        print(f"Parameters: structure_type={structure_type}, material={material}, size={size}")
        
        # Run simulation
        fig = run_ase_simulation(
            structure_type=structure_type,
            material=material,
            size=size,
            vacuum=vacuum,
            temperature=temperature,
            timestep=timestep,
            steps=steps
        )
        
        print("ASE simulation completed successfully")
        return jsonify({
            'plot': fig.to_json()
        })
        
    except Exception as e:
        print(f"Error in ASE route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/pyvista', methods=['POST'])
def pyvista():
    try:
        data = request.get_json()
        print("Received PyVista request with data:", data)
        
        # Get visualization parameters
        particle_type = data.get('particle_type', 'sphere')
        
        # Get particle-specific parameters
        params = {}
        if particle_type == 'sphere':
            params['radius'] = float(data.get('radius', 10))
            params['resolution'] = int(data.get('resolution', 30))
        elif particle_type == 'rod':
            params['length'] = float(data.get('length', 20))
            params['radius'] = float(data.get('radius', 5))
            params['resolution'] = int(data.get('resolution', 30))
        elif particle_type == 'cube':
            params['size'] = float(data.get('size', 10))
        elif particle_type == 'core-shell':
            params['core_radius'] = float(data.get('core_radius', 8))
            params['shell_thickness'] = float(data.get('shell_thickness', 2))
            params['resolution'] = int(data.get('resolution', 30))
            
        # Get common parameters
        params['add_ligands'] = bool(data.get('add_ligands', False))
        if params['add_ligands']:
            params['n_ligands'] = int(data.get('n_ligands', 10))
            params['ligand_length'] = float(data.get('ligand_length', 2))
            
        print(f"Parameters: particle_type={particle_type}, params={params}")
        
        # Run visualization
        fig = run_pyvista_visualization(particle_type, **params)
        
        print("PyVista visualization completed successfully")
        return jsonify({
            'plot': fig.to_json()
        })
        
    except Exception as e:
        print(f"Error in PyVista route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/pymatgen', methods=['POST'])
def pymatgen():
    try:
        data = request.get_json()
        print("Received pymatgen request with data:", data)
        
        # Get analysis parameters
        material_type = data.get('material_type', 'bulk')
        analysis_type = data.get('analysis_type', 'structure')
        
        # Get material-specific parameters
        params = {}
        if material_type == 'bulk':
            params['lattice_constant'] = float(data.get('lattice_constant', 3.5))
            params['species'] = data.get('species', ['Au'])
        elif material_type == 'surface':
            params['miller_index'] = tuple(data.get('miller_index', [1, 1, 1]))
            params['min_slab_size'] = float(data.get('min_slab_size', 10.0))
            params['min_vacuum_size'] = float(data.get('min_vacuum_size', 10.0))
        elif material_type == 'nanoparticle':
            params['radius'] = float(data.get('radius', 10.0))
            
        # Get analysis-specific parameters
        if analysis_type == 'bonding':
            params['method'] = data.get('method', 'voronoi')
            
        print(f"Parameters: material_type={material_type}, analysis_type={analysis_type}, params={params}")
        
        # Run analysis
        fig = run_pymatgen_analysis(
            material_type=material_type,
            analysis_type=analysis_type,
            **params
        )
        
        print("Pymatgen analysis completed successfully")
        return jsonify({
            'plot': fig.to_json()
        })
        
    except Exception as e:
        print(f"Error in pymatgen route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/qe', methods=['POST'])
def qe():
    try:
        data = request.get_json()
        print("Received QE request with data:", data)
        
        # Get simulation parameters
        structure_type = data.get('structure_type', 'bulk')
        calculation_type = data.get('calculation_type', 'scf')
        plot_type = data.get('plot_type', 'structure')
        
        # Get structure-specific parameters
        params = {}
        if structure_type == 'bulk':
            params['lattice_constant'] = float(data.get('lattice_constant', 4.0))
        elif structure_type == 'surface':
            params['layers'] = int(data.get('layers', 4))
            params['vacuum'] = float(data.get('vacuum', 10.0))
        elif structure_type == 'nanoparticle':
            params['size'] = tuple(data.get('size', [3, 3, 3]))
            
        # Get calculation parameters
        if calculation_type == 'scf':
            params['kpts'] = tuple(data.get('kpts', [4, 4, 4]))
            params['ecutwfc'] = float(data.get('ecutwfc', 40.0))
            params['ecutrho'] = float(data.get('ecutrho', 320.0))
        elif calculation_type == 'bands':
            params['kpts'] = tuple(data.get('kpts', [8, 8, 8]))
            params['bands_points'] = int(data.get('bands_points', 40))
        elif calculation_type == 'dos':
            params['kpts'] = tuple(data.get('kpts', [12, 12, 12]))
            params['width'] = float(data.get('width', 0.2))
            
        print(f"Parameters: structure_type={structure_type}, calculation_type={calculation_type}, params={params}")
        
        # Run simulation
        fig = run_qe_simulation(
            structure_type=structure_type,
            calculation_type=calculation_type,
            plot_type=plot_type,
            **params
        )
        
        print("QE simulation completed successfully")
        return jsonify({
            'plot': fig.to_json()
        })
        
    except Exception as e:
        print(f"Error in QE route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/chemml', methods=['POST'])
def chemml():
    try:
        data = request.get_json()
        print("Received ChemML request with data:", data)
        
        # Get analysis parameters
        analysis_type = data.get('analysis_type', 'training')
        
        # Run analysis
        fig = run_chemml_analysis(analysis_type, **data)
        
        response = {
            'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        # Add molecular descriptors if available
        if hasattr(fig, 'descriptors'):
            response['descriptors'] = fig.descriptors
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Error in ChemML analysis: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)