from flask import Flask, request, jsonify, render_template
from chembl_webresource_client.new_client import new_client

app = Flask(__name__)

class MoleculeIdentifier:
    def __init__(self):
        # Initialize molecule client
        self.molecule_client = new_client.molecule

    def get_detailed_molecule_info(self, chembl_id):
        """
        Retrieve detailed information for a specific ChEMBL molecule
        
        Args:
            chembl_id (str): ChEMBL identifier for the molecule
        
        Returns:
            dict: Comprehensive molecule details
        """
        try:
            # Fetch full molecule details
            molecule = self.molecule_client.get(chembl_id)
            
            if not molecule:
                return None
            
            # Extract properties with comprehensive details
            mol_props = molecule.get('molecule_properties', {}) or {}
            mol_structures = molecule.get('molecule_structures', {}) or {}
            
            detailed_info = {
                'ChEMBL ID': molecule.get('molecule_chembl_id', 'N/A'),
                'Preferred Name': molecule.get('pref_name', 'N/A'),
                'Molecule Type': molecule.get('molecule_type', 'N/A'),
                
                # Structural Properties
                'Molecular Weight': mol_props.get('molecular_weight', 'N/A'),
                'Molecular Formula': mol_props.get('molecular_formula', 'N/A'),
                'ALogP': mol_props.get('alogp', 'N/A'),
                'Molecular Species': mol_props.get('molecular_species', 'N/A'),
                
                # Structural Representations
                'Canonical SMILES': mol_structures.get('canonical_smiles', 'N/A'),
                'Standard InChI': mol_structures.get('standard_inchi', 'N/A'),
                'Standard InChI Key': mol_structures.get('standard_inchi_key', 'N/A'),
                
                # Additional Metadata
                'First Approval': molecule.get('first_approval', 'N/A'),
                'Max Phase': molecule.get('max_phase', 'N/A'),
                'Oral': mol_props.get('ro5_violations', 'N/A'),
                'Molecular Complexity': mol_props.get('molecular_species', 'N/A')
            }
            
            return detailed_info
        
        except Exception as e:
            print(f"Error retrieving detailed molecule info: {e}")
            return None

    def get_chembl_ids_from_smiles(self, smiles):
        """
        Retrieve ChEMBL IDs for a given SMILES string
        
        Args:
            smiles (str): SMILES representation of the molecule
        
        Returns:
            list: Detailed information for molecules matching the SMILES
        """
        try:
            # Use the molecule client to search by SMILES
            results = self.molecule_client.filter(molecule_structures__canonical_smiles__flexmatch=smiles)
            
            # Extract and enrich ChEMBL matches with detailed info
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

# Initialize molecule identifier
molecule_identifier = MoleculeIdentifier()

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route to handle SMILES input and display results
    """
    results = None
    smiles = None
    
    if request.method == 'POST':
        # Get SMILES from form submission
        smiles = request.form.get('smiles', '').strip()
        
        if smiles:
            # Perform SMILES lookup
            results = molecule_identifier.get_chembl_ids_from_smiles(smiles)
    
    return render_template('index.html', results=results, smiles=smiles)

@app.route('/api/molecule', methods=['GET'])
def molecule_api():
    """
    API endpoint for SMILES lookup
    """
    # Get SMILES from query parameter
    smiles = request.args.get('smiles', '').strip()
    
    if not smiles:
        return jsonify({
            'error': 'No SMILES string provided',
            'status': 'error'
        }), 400
    
    # Perform SMILES lookup
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

# Error handler for 404 Not Found
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# Main block
if __name__ == '__main__':
    app.run(debug=True)
