from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly
import json
from quantum_orbitals import (
    create_atom, calculate_orbital_wavefunction, plot_orbital_density,
    get_available_atoms, get_available_basis_sets
)
from ase_simulation import run_ase_simulation
from pymatgen_analysis import run_pymatgen_analysis
from chemml_analysis import run_chemml_analysis
from openbabel_analysis import run_openbabel_analysis, compare_molecules, analyze_reaction
from mdanalysis_simulation import run_basic_analysis
from werkzeug.utils import secure_filename
import os
from chembl_webresource_client.new_client import new_client

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class MoleculeIdentifier:
    def __init__(self):
        # Initialize molecule client
        self.molecule_client = new_client.molecule

    def get_detailed_molecule_info(self, chembl_id):
        """
        Retrieve detailed information for a specific ChEMBL molecule
        """
        try:
            # Fetch full molecule details
            molecule = self.molecule_client.get(chembl_id)
            
            if not molecule:
                return None
            
            # Extract all available properties
            mol_props = molecule.get('molecule_properties', {}) or {}
            mol_structures = molecule.get('molecule_structures', {}) or {}
            
            detailed_info = {
                # Basic Information
                'ChEMBL ID': molecule.get('molecule_chembl_id', 'N/A'),
                'Preferred Name': molecule.get('pref_name', 'N/A'),
                'Molecule Type': molecule.get('molecule_type', 'N/A'),
                'Max Phase': molecule.get('max_phase', 'N/A'),
                'First Approval': molecule.get('first_approval', 'N/A'),
                'Oral': molecule.get('oral', 'N/A'),
                'Parenteral': molecule.get('parenteral', 'N/A'),
                'Topical': molecule.get('topical', 'N/A'),
                'Black Box Warning': molecule.get('black_box_warning', 'N/A'),
                'Natural Product': molecule.get('natural_product', 'N/A'),
                'First in Class': molecule.get('first_in_class', 'N/A'),
                'Chirality': molecule.get('chirality', 'N/A'),
                'Prodrug': molecule.get('prodrug', 'N/A'),
                'Inorganic Flag': molecule.get('inorganic_flag', 'N/A'),
                
                # Structural Properties
                'Molecular Formula': mol_props.get('full_molformula', 'N/A'),
                'Molecular Weight': mol_props.get('full_mwt', 'N/A'),
                'ALOGP': mol_props.get('alogp', 'N/A'),
                'RTB': mol_props.get('rtb', 'N/A'),
                'PSA': mol_props.get('psa', 'N/A'),
                'HBA': mol_props.get('hba', 'N/A'),
                'HBD': mol_props.get('hbd', 'N/A'),
                'Heavy Atoms': mol_props.get('heavy_atoms', 'N/A'),
                'Aromatic Rings': mol_props.get('aromatic_rings', 'N/A'),
                'Structure Type': mol_props.get('structure_type', 'N/A'),
                
                # Drug-likeness Properties
                'QED Weighted': mol_props.get('qed_weighted', 'N/A'),
                'CX LogP': mol_props.get('cx_logp', 'N/A'),
                'CX LogD': mol_props.get('cx_logd', 'N/A'),
                'Molecular Species': mol_props.get('molecular_species', 'N/A'),
                'Ro3 Pass': mol_props.get('ro3_pass', 'N/A'),
                'Ro5 Pass': mol_props.get('num_ro5_violations', 'N/A'),
                
                # Structural Representations
                'Canonical SMILES': mol_structures.get('canonical_smiles', 'N/A'),
                'Standard InChI': mol_structures.get('standard_inchi', 'N/A'),
                'Standard InChI Key': mol_structures.get('standard_inchi_key', 'N/A'),
                'MOLFILE': mol_structures.get('molfile', 'N/A'),
                
                # Additional Properties
                'Availability Type': molecule.get('availability_type', 'N/A'),
                'Cross References': molecule.get('cross_references', 'N/A'),
                'Synonyms': molecule.get('molecule_synonyms', []),
                'Helm Notation': molecule.get('helm_notation', 'N/A'),
                'Biotherapeutic': molecule.get('biotherapeutic', {}),
                'Withdrawn Flag': molecule.get('withdrawn_flag', 'N/A'),
                'Withdrawn Reason': molecule.get('withdrawn_reason', 'N/A'),
                'Withdrawn Country': molecule.get('withdrawn_country', 'N/A'),
                'Withdrawn Year': molecule.get('withdrawn_year', 'N/A')
            }
            
            return detailed_info
        
        except Exception as e:
            print(f"Error retrieving detailed molecule info: {e}")
            return None

    def get_chembl_ids_from_smiles(self, smiles):
        """
        Retrieve ChEMBL IDs for a given SMILES string
        """
        try:
            results = self.molecule_client.filter(molecule_structures__canonical_smiles__flexmatch=smiles)
            detailed_matches = []
            for mol in results:
                chembl_id = mol.get('molecule_chembl_id')
                detailed_info = self.get_detailed_molecule_info(chembl_id)
                if detailed_info:
                    detailed_matches.append(detailed_info)
            return detailed_matches
        except Exception as e:
            print(f"Error searching for SMILES: {e}")
            return []

# Initialize molecule identifier after app creation
molecule_identifier = MoleculeIdentifier()

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

@app.route('/ase', methods=['POST'])
def ase():
    try:
        data = request.get_json()
        print("Received ASE request with data:", data)
        
        # Get simulation parameters
        structure_type = data.get('structure_type', 'bulk')
        material = data.get('material', 'Cu')
        vacuum = float(data.get('vacuum', 10.0))
        temperature = float(data.get('temperature', 300))
        timestep = float(data.get('timestep', 1.0))
        steps = int(data.get('steps', 100))
        
        print(f"Parameters: structure_type={structure_type}, material={material}")
        
        # Run simulation
        fig = run_ase_simulation(
            structure_type=structure_type,
            material=material,
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
        
        # Run analysis with data directly
        result = run_chemml_analysis(**data)
        
        response = {
            'plot': json.dumps(result['plot'], cls=plotly.utils.PlotlyJSONEncoder)
        }
        
        # Add molecular descriptors if available
        if result['properties']:
            response['properties'] = result['properties']
        
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
    temp_file = None
    try:
        # Check if a file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and file.filename.endswith('.pdb'):
            # Secure the filename and save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            try:
                # Save the uploaded file
                file.save(filepath)
                temp_file = filepath  # Store filepath for cleanup
                
                # Get analysis options from request
                options = {
                    'analyze_secondary': request.form.get('analyze_secondary', 'true').lower() == 'true',
                    'analyze_contacts': request.form.get('analyze_contacts', 'true').lower() == 'true',
                    'analyze_rmsd': request.form.get('analyze_rmsd', 'true').lower() == 'true',
                    'contact_cutoff': float(request.form.get('contact_cutoff', 6.0)),
                    'show_backbone': request.form.get('show_backbone', 'true').lower() == 'true',
                    'show_sidechains': request.form.get('show_sidechains', 'false').lower() == 'true',
                    'show_hydrogens': request.form.get('show_hydrogens', 'false').lower() == 'true',
                    'color_scheme': request.form.get('color_scheme', 'element')
                }
                
                # Run the analysis with options
                results = run_basic_analysis(filepath, options)
                
                # Convert plots to JSON
                if results.get('plots'):
                    results['plots'] = [plot.to_json() for plot in results['plots']]
                
                return jsonify(results)
                
            finally:
                # Clean up the uploaded file
                if temp_file and os.path.exists(temp_file):
                    try:
                        file.close()
                        # Close any open file handles
                        import psutil
                        process = psutil.Process()
                        for handler in process.open_files():
                            if handler.path == temp_file:
                                os.close(handler.fd)
                        os.remove(temp_file)
                    except Exception as e:
                        print(f"Warning: Could not remove temporary file: {str(e)}")
        
        return jsonify({'error': 'Invalid file type'}), 400
        
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
        similarity, description, fig = compare_molecules(smiles1, smiles2, fp_type)
        
        if similarity is None:
            raise ValueError("Failed to calculate similarity")
            
        return jsonify({
            'similarity': similarity,
            'description': description,
            'plot': fig.to_json() if fig else None
        })
        
    except Exception as e:
        print(f"Error in similarity calculation: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/openbabel/reaction', methods=['POST'])
def openbabel_reaction():
    try:
        data = request.get_json()
        print("Received reaction analysis request with data:", data)
        
        # Get reaction SMILES and validation options
        reaction_smiles = data.get('reaction_smiles')
        validation_options = data.get('validation_options', {})
        
        if not reaction_smiles:
            raise ValueError("Reaction SMILES is required")
            
        # Analyze reaction
        result = analyze_reaction(reaction_smiles, validation_options)
        
        if result is None:
            raise ValueError("Failed to analyze reaction")
            
        return jsonify({
            'validation_html': result['validation_html'],
            'plot': result['plot'].to_json() if result['plot'] else None,
            'is_valid': result['is_valid']
        })
        
    except Exception as e:
        print(f"Error in reaction analysis: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/chembl', methods=['GET', 'POST'])
def chembl():
    """
    Route to handle SMILES input and display results
    """
    results = None
    smiles = None
    
    if request.method == 'POST':
        # Get SMILES from form submission
        smiles = request.form.get('smiles', '').strip()
        
        if smiles:
            # Perform SMILES lookup
            results = molecule_identifier.get_chembl_ids_from_smiles(smiles)
    
    return render_template('chembl.html', results=results, smiles=smiles, embedded=request.args.get('embedded', False))

@app.route('/api/molecule', methods=['GET'])
def molecule_api():
    """
    API endpoint for SMILES lookup
    """
    smiles = request.args.get('smiles', '').strip()
    
    if not smiles:
        return jsonify({
            'error': 'No SMILES string provided',
            'status': 'error'
        }), 400
    
    results = molecule_identifier.get_chembl_ids_from_smiles(smiles)
    
    if results:
        return jsonify({
            'status': 'success',
            'smiles': smiles,
            'matches': results
        })
    else:
        return jsonify({
            'status': 'no_matches',
            'smiles': smiles,
            'message': 'No ChEMBL molecules found for the given SMILES string'
        }), 404

if __name__ == '__main__':
    # Development server
    app.run(host='0.0.0.0', port=5000, debug=True)
else:
    # Production server (gunicorn)
    # Configure any production-specific settings
    app.config['DEBUG'] = False
    app.config['TEMPLATES_AUTO_RELOAD'] = False