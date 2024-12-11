import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openbabel import openbabel as ob
from openbabel import pybel
import tempfile
import os

class MoleculeHandler:
    def __init__(self):
        self.mol = None
        self.format = None
        self.temp_dir = tempfile.mkdtemp()
        
    def read_molecule(self, data, input_format='smiles'):
        """Read molecular data in various formats and generate 3D coordinates."""
        try:
            if input_format == 'smiles':
                # Read SMILES and create molecule
                self.mol = pybel.readstring('smi', data)
                
                # Generate 3D coordinates
                self.mol.make3D(forcefield='mmff94', steps=100)
                print("Generated 3D coordinates")
                
            elif input_format == 'xyz':
                self.mol = pybel.readstring('xyz', data)
            elif input_format == 'pdb':
                self.mol = pybel.readstring('pdb', data)
            else:
                raise ValueError(f"Unsupported input format: {input_format}")
                
            self.format = input_format
            return True
        except Exception as e:
            print(f"Error reading molecule: {str(e)}")
            return False
            
    def convert_format(self, output_format='xyz'):
        """Convert molecule to different format."""
        if self.mol is None:
            return None
            
        try:
            return self.mol.write(output_format)
        except Exception as e:
            print(f"Error converting format: {str(e)}")
            return None
            
    def optimize_geometry(self, force_field='mmff94', steps=500):
        """Optimize molecular geometry."""
        if self.mol is None:
            return False
            
        try:
            # Set up force field
            if force_field == 'mmff94':
                ff = pybel._forcefields["mmff94"]
            elif force_field == 'uff':
                ff = pybel._forcefields["uff"]
            else:
                raise ValueError(f"Unsupported force field: {force_field}")
                
            # Setup force field for molecule
            if not ff.Setup(self.mol.OBMol):
                print("Could not setup force field")
                return False
                
            # Optimize geometry
            ff.ConjugateGradients(steps)
            ff.GetCoordinates(self.mol.OBMol)
            
            return True
        except Exception as e:
            print(f"Error optimizing geometry: {str(e)}")
            return False
            
    def calculate_descriptors(self):
        """Calculate molecular descriptors."""
        if self.mol is None:
            return None
            
        try:
            # Count bonds using OBMol
            n_bonds = self.mol.OBMol.NumBonds()
            
            # Basic descriptors
            descriptors = {
                'molecular_weight': self.mol.molwt,
                'exact_mass': self.mol.exactmass,
                'formula': self.mol.formula,
                'n_atoms': len(self.mol.atoms),
                'n_bonds': n_bonds,
                'n_rotatable_bonds': len([b for b in ob.OBMolBondIter(self.mol.OBMol) if b.IsRotor()]),
                'hbd': len([a for a in self.mol.atoms if a.OBAtom.IsHbondDonor()]),
                'hba': len([a for a in self.mol.atoms if a.OBAtom.IsHbondAcceptor()])
            }
            
            # Calculate additional descriptors for predictions
            mol = self.mol.OBMol
            
            # Calculate LogP using built-in method
            logp = 0
            try:
                logp_filter = ob.OBDescriptor.FindType("logp")
                if logp_filter is not None:
                    logp = logp_filter.Predict(mol)
            except:
                print("Warning: Could not calculate LogP")
                
            # Calculate TPSA using built-in method
            tpsa = 0
            try:
                tpsa_filter = ob.OBDescriptor.FindType("TPSA")
                if tpsa_filter is not None:
                    tpsa = tpsa_filter.Predict(mol)
            except:
                print("Warning: Could not calculate TPSA")
                
            # Add calculated descriptors
            descriptors.update({
                'topological_polar_surface_area': tpsa,
                'logp': logp,
                'aromatic_rings': len([r for r in ob.OBMolRingIter(mol) if r.IsAromatic()]),
                'heavy_atoms': mol.NumHvyAtoms()
            })
            
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors: {str(e)}")
            return None
            
    def predict_properties(self, descriptors):
        """Predict molecular properties using descriptors."""
        if descriptors is None:
            return None
            
        try:
            # Get LogP value, default to estimated value if not available
            logp = descriptors.get('logp', 0)
            if logp == 0:
                # Estimate LogP based on heavy atoms and aromatic rings
                logp = (descriptors['heavy_atoms'] * 0.2 + 
                       descriptors['aromatic_rings'] * 0.5)
            
            # Estimate solubility using modified Yalkowsky equation
            # logS = 0.5 - 0.01 * (MP - 25) - logP
            # Using estimated melting point based on molecular weight
            est_mp = 0.1 * descriptors['molecular_weight'] + 100
            solubility = 0.5 - 0.01 * (est_mp - 25) - logp
            
            # Calculate Lipinski's Rule of 5 parameters
            lipinski = {
                'molecular_weight_ok': descriptors['molecular_weight'] <= 500,
                'logp_ok': -0.4 <= logp <= 5.6,
                'hbd_ok': descriptors['hbd'] <= 5,
                'hba_ok': descriptors['hba'] <= 10,
                'rotatable_bonds_ok': descriptors['n_rotatable_bonds'] <= 10
            }
            
            # Calculate bioavailability score (0-1)
            bioavailability = sum(1 for v in lipinski.values() if v) / len(lipinski)
            
            # Estimate toxicity risk based on structural features
            toxicity_risk = self._estimate_toxicity(descriptors)
            
            # Estimate biological activity based on physicochemical properties
            activity_score = self._estimate_biological_activity(descriptors)
            
            predictions = {
                'logp': logp,
                'solubility': solubility,
                'bioavailability': bioavailability,
                'lipinski_rules': lipinski,
                'toxicity_risk': toxicity_risk,
                'activity_score': activity_score
            }
            
            return predictions
        except Exception as e:
            print(f"Error predicting properties: {str(e)}")
            return None
            
    def _estimate_toxicity(self, descriptors):
        """Simple toxicity estimation based on molecular features."""
        risk_score = 0
        
        # High molecular weight increases risk
        if descriptors['molecular_weight'] > 800:
            risk_score += 1
            
        # Very high or low LogP increases risk
        if descriptors['logp'] > 5 or descriptors['logp'] < -2:
            risk_score += 1
            
        # High number of aromatic rings increases risk
        if descriptors['aromatic_rings'] > 3:
            risk_score += 1
            
        # High TPSA might indicate permeability issues
        if descriptors['topological_polar_surface_area'] > 140:
            risk_score += 1
            
        return {
            'score': risk_score,
            'level': 'Low' if risk_score <= 1 else 'Medium' if risk_score <= 2 else 'High'
        }
        
    def _estimate_biological_activity(self, descriptors):
        """Estimate potential biological activity based on drug-likeness rules."""
        score = 0
        max_score = 5
        
        # Molecular weight between 160 and 500
        if 160 <= descriptors['molecular_weight'] <= 500:
            score += 1
            
        # LogP between -0.4 and 5.6 (extended Lipinski range)
        if -0.4 <= descriptors['logp'] <= 5.6:
            score += 1
            
        # Topological polar surface area between 20 and 140
        if 20 <= descriptors['topological_polar_surface_area'] <= 140:
            score += 1
            
        # Number of rotatable bonds <= 10
        if descriptors['n_rotatable_bonds'] <= 10:
            score += 1
            
        # Hydrogen bond donors and acceptors within range
        if descriptors['hbd'] <= 5 and descriptors['hba'] <= 10:
            score += 1
            
        return {
            'score': score / max_score,
            'level': 'Low' if score <= 2 else 'Medium' if score <= 3 else 'High'
        }
        
    def create_visualization(self, plot_type='3d'):
        """Create visualization of molecular structure and predictions."""
        if self.mol is None:
            return None
            
        try:
            # Calculate descriptors and predictions
            descriptors = self.calculate_descriptors()
            predictions = self.predict_properties(descriptors)
            
            # Create subplots: 3D structure and predictions
            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'scatter3d'}, {'type': 'table'}]],
                subplot_titles=('Molecular Structure', 'Properties & Predictions')
            )
            
            # Add 3D structure visualization
            coords = np.array([[a.coords[i] for i in range(3)] for a in self.mol.atoms])
            elements = [a.type for a in self.mol.atoms]
            
            # Element-color mapping
            element_colors = {
                'C': 'gray', 'H': 'white', 'O': 'red', 'N': 'blue',
                'S': 'yellow', 'P': 'orange', 'F': 'green', 'Cl': 'green',
                'Br': 'brown', 'I': 'purple'
            }
            
            # Plot atoms
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[element_colors.get(el, 'gray') for el in elements],
                        symbol='circle'
                    ),
                    text=elements,
                    name='Atoms'
                ),
                row=1, col=1
            )
            
            # Add bonds
            for bond in ob.OBMolBondIter(self.mol.OBMol):
                begin_idx = bond.GetBeginAtomIdx() - 1
                end_idx = bond.GetEndAtomIdx() - 1
                begin_coords = coords[begin_idx]
                end_coords = coords[end_idx]
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[begin_coords[0], end_coords[0]],
                        y=[begin_coords[1], end_coords[1]],
                        z=[begin_coords[2], end_coords[2]],
                        mode='lines',
                        line=dict(color='gray', width=2),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Create property table
            if descriptors and predictions:
                # Combine basic properties and predictions
                table_data = {
                    'Property': [],
                    'Value': []
                }
                
                # Basic properties
                table_data['Property'].extend([
                    'Formula', 'Molecular Weight', 'Number of Atoms',
                    'Number of Bonds', 'Number of Rotatable Bonds'
                ])
                table_data['Value'].extend([
                    descriptors['formula'],
                    f"{descriptors['molecular_weight']:.2f}",
                    str(descriptors['n_atoms']),
                    str(descriptors['n_bonds']),
                    str(descriptors['n_rotatable_bonds'])
                ])
                
                # Predictions
                table_data['Property'].extend([
                    'LogP', 'Solubility (logS)', 'Bioavailability Score',
                    'Toxicity Risk', 'Biological Activity'
                ])
                table_data['Value'].extend([
                    f"{predictions['logp']:.2f}",
                    f"{predictions['solubility']:.2f}",
                    f"{predictions['bioavailability']:.2f}",
                    predictions['toxicity_risk']['level'],
                    predictions['activity_score']['level']
                ])
                
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=list(table_data.keys()),
                            fill_color='paleturquoise',
                            align='left'
                        ),
                        cells=dict(
                            values=list(table_data.values()),
                            fill_color='lavender',
                            align='left'
                        )
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Molecular Analysis with Predictions',
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)'
                ),
                height=800,
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None

def run_openbabel_analysis(input_format='smiles', data=None, **kwargs):
    """Run molecular analysis using OpenBabel."""
    try:
        # Validate input data
        if not data or data.strip() == '':
            # If no data provided, use an example molecule
            data = 'CC(=O)O'  # Acetic acid
            print(f"No input provided. Using example molecule: {data}")
            
        print(f"Starting OpenBabel analysis with format: {input_format}")
        
        # Initialize handler
        handler = MoleculeHandler()
        
        # Read molecule
        if not handler.read_molecule(data, input_format):
            # Create error figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="Failed to read molecular structure. Please check your input.",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Error: Invalid Input',
                height=400
            )
            return fig
            
        print("Molecule read successfully")
        
        # Optimize geometry if requested
        if kwargs.get('optimize', False):
            force_field = kwargs.get('force_field', 'mmff94')
            steps = kwargs.get('steps', 500)
            if handler.optimize_geometry(force_field, steps):
                print("Geometry optimized")
            else:
                print("Geometry optimization failed")
        
        # Create visualization
        fig = handler.create_visualization()
        print("Visualization created")
        
        return fig
        
    except Exception as e:
        print(f"Error in OpenBabel analysis: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title='Error',
            height=400
        )
        return fig 