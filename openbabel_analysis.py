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
            
            # Only include descriptors that don't require contribution data files
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
            return descriptors
        except Exception as e:
            print(f"Error calculating descriptors: {str(e)}")
            return None
            
    def create_visualization(self, plot_type='3d'):
        """Create visualization of molecular structure."""
        if self.mol is None:
            return None
            
        try:
            # Get atomic coordinates
            coords = np.array([[a.coords[i] for i in range(3)] for a in self.mol.atoms])
            elements = [a.type for a in self.mol.atoms]
            
            # Create element-color mapping
            element_colors = {
                'C': 'gray',
                'H': 'white',
                'O': 'red',
                'N': 'blue',
                'S': 'yellow',
                'P': 'orange',
                'F': 'green',
                'Cl': 'green',
                'Br': 'brown',
                'I': 'purple'
            }
            
            fig = make_subplots(rows=1, cols=2,
                               subplot_titles=('Structure', 'Properties'),
                               specs=[[{'type': 'scatter3d'}, {'type': 'table'}]])
            
            # Plot 3D structure
            fig.add_trace(
                go.Scatter3d(
                    x=coords[:, 0],
                    y=coords[:, 1],
                    z=coords[:, 2],
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
                begin_idx = bond.GetBeginAtomIdx() - 1  # OpenBabel uses 1-based indexing
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
            
            # Calculate and display descriptors
            descriptors = self.calculate_descriptors()
            if descriptors:
                fig.add_trace(
                    go.Table(
                        header=dict(
                            values=['Property', 'Value'],
                            fill_color='paleturquoise',
                            align='left'
                        ),
                        cells=dict(
                            values=[
                                list(descriptors.keys()),
                                list(descriptors.values())
                            ],
                            fill_color='lavender',
                            align='left'
                        )
                    ),
                    row=1, col=2
                )
            
            # Update layout
            fig.update_layout(
                title='Molecular Structure Analysis',
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)'
                ),
                height=600,
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