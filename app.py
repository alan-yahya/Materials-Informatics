from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly
import json
from quantum_orbitals import (
    create_atom, calculate_orbital_wavefunction, plot_orbital_density,
    get_available_atoms, get_available_basis_sets
)
from battery_simulation import run_battery_simulation
from double_slit import run_double_slit_simulation
from semiclassical import run_semiclassical_simulation
from pic_simulation import run_pic_simulation
from shape_optimization import run_shape_optimization
from qutip_simulation import run_qutip_simulation
from ase_simulation import run_ase_simulation
from pymatgen_analysis import run_pymatgen_analysis
from chemml_analysis import run_chemml_analysis
from openbabel_analysis import run_openbabel_analysis, compare_molecules
from mdanalysis_simulation import run_mdanalysis

app = Flask(__name__)

@app.route('/')
def index():
    atoms = get_available_atoms()
    basis_sets = get_available_basis_sets()
    return render_template('index.html', atoms=atoms, basis_sets=basis_sets)

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

@app.route('/openbabel', methods=['POST'])
def openbabel():
    try:
        data = request.get_json()
        print("Received OpenBabel request with data:", data)
        
        # Run analysis
        fig, pdb_data = run_openbabel_analysis(
            input_format=data.get('input_format', 'smiles'),
            data=data.get('data', ''),
            optimize=data.get('optimize', False),
            force_field=data.get('force_field', 'mmff94'),
            steps=data.get('steps', 500)
        )
        
        return jsonify({
            'plot': fig.to_json(),
            'pdb_data': pdb_data
        })
        
    except Exception as e:
        print(f"Error in OpenBabel route: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/mdanalysis', methods=['POST'])
def mdanalysis():
    try:
        data = request.get_json()
        print("Received MDAnalysis request with data:", data)
        
        # Extract parameters from data
        params = {
            'topology_data': data.get('topology_data'),
            'trajectory_data': data.get('trajectory_data'),
            'analysis_type': data.get('analysis_type', 'structure'),
            'selection': data.get('selection', 'all'),
            'topology_format': data.get('topology_format', 'pdb'),
            'trajectory_format': data.get('trajectory_format', 'none')
        }
        
        if not params['topology_data']:
            raise ValueError("No topology data provided")
            
        # Run analysis
        fig = run_mdanalysis(**params)
        
        return jsonify({
            'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        })
        
    except Exception as e:
        print(f"Error in MDAnalysis: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/openbabel/similarity', methods=['POST'])
def openbabel_similarity():
    try:
        data = request.get_json()
        print("Received similarity request with data:", data)
        
        # Get SMILES strings and fingerprint type
        smiles1 = data.get('smiles1')
        smiles2 = data.get('smiles2')
        fp_type = data.get('fp_type', 'fp2')
        
        if not smiles1 or not smiles2:
            raise ValueError("Both SMILES strings are required")
            
        # Calculate similarity and get visualization
        similarity, fig = compare_molecules(smiles1, smiles2, fp_type)
        
        if similarity is None:
            raise ValueError("Failed to calculate similarity")
            
        return jsonify({
            'similarity': similarity,
            'plot': fig.to_json() if fig else None
        })
        
    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(debug=True)