import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, Descriptors, rdDepictor

class MoleculeVisualizer:
    def __init__(self):
        self.mol = None
        self.properties = {}
        
    def load_molecule(self, input_data, input_type='smiles'):
        """Load molecule from SMILES or MOL format."""
        try:
            if input_type == 'smiles':
                self.mol = Chem.MolFromSmiles(input_data)
                if self.mol is not None:
                    self.mol = Chem.AddHs(self.mol)
            elif input_type == 'mol':
                self.mol = Chem.MolFromMolBlock(input_data)
                if self.mol is not None:
                    self.mol = Chem.AddHs(self.mol)
            else:
                raise ValueError(f"Unsupported input type: {input_type}")
                
            if self.mol is None:
                raise ValueError("Failed to parse molecule")
                
            # Generate 3D coordinates if not present
            if not self.mol.GetNumConformers():
                rdDepictor.Compute2DCoords(self.mol)
                AllChem.EmbedMolecule(self.mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(self.mol)
                
            return True
        except Exception as e:
            print(f"Error loading molecule: {str(e)}")
            return False
            
    def calculate_properties(self):
        """Calculate molecular properties."""
        if self.mol is None:
            return None
            
        try:
            self.properties = {
                'Molecular Weight': round(Descriptors.ExactMolWt(self.mol), 2),
                'LogP': round(Descriptors.MolLogP(self.mol), 2),
                'H-Bond Donors': Descriptors.NumHDonors(self.mol),
                'H-Bond Acceptors': Descriptors.NumHAcceptors(self.mol),
                'Rotatable Bonds': Descriptors.NumRotatableBonds(self.mol),
                'Ring Count': Descriptors.RingCount(self.mol),
                'Aromatic Ring Count': Descriptors.NumAromaticRings(self.mol),
                'Topological Polar Surface Area': round(Descriptors.TPSA(self.mol), 2),
                'Formula': Chem.rdMolDescriptors.CalcMolFormula(self.mol)
            }
            return self.properties
        except Exception as e:
            print(f"Error calculating properties: {str(e)}")
            return None
            
    def create_visualization(self, viz_type='2d'):
        """Create molecular visualization."""
        if self.mol is None:
            return None
            
        try:
            if viz_type == '2d':
                return self._create_2d_plot()
            elif viz_type == '3d':
                return self._create_3d_plot()
            elif viz_type == 'descriptors':
                return self._create_descriptor_plot()
            else:
                raise ValueError(f"Unsupported visualization type: {viz_type}")
                
        except Exception as e:
            print(f"Error creating visualization: {str(e)}")
            return None
            
    def _create_2d_plot(self):
        """Create 2D structure plot."""
        img = Draw.MolToImage(self.mol)
        
        # Convert PIL image to numpy array
        img_array = np.array(img)
        
        # Create figure
        fig = go.Figure(data=[go.Image(z=img_array)])
        
        # Update layout
        fig.update_layout(
            title='2D Structure',
            xaxis_visible=False,
            yaxis_visible=False,
            width=500,
            height=500
        )
        
        return fig
        
    def _create_3d_plot(self):
        """Create 3D structure plot."""
        # Get conformer
        conf = self.mol.GetConformer()
        
        # Create figure
        fig = go.Figure()
        
        # Add atoms
        positions = conf.GetPositions()
        atoms = [atom.GetSymbol() for atom in self.mol.GetAtoms()]
        
        # Color mapping
        color_map = {
            'C': 'gray', 'N': 'blue', 'O': 'red', 'S': 'yellow',
            'F': 'green', 'Cl': 'green', 'Br': 'brown', 'I': 'purple',
            'P': 'orange', 'B': 'pink'
        }
        
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=[color_map.get(atom, 'gray') for atom in atoms],
                symbol='circle'
            ),
            text=atoms,
            name='Atoms'
        ))
        
        # Add bonds
        for bond in self.mol.GetBonds():
            id1 = bond.GetBeginAtomIdx()
            id2 = bond.GetEndAtomIdx()
            
            fig.add_trace(go.Scatter3d(
                x=[positions[id1, 0], positions[id2, 0]],
                y=[positions[id1, 1], positions[id2, 1]],
                z=[positions[id1, 2], positions[id2, 2]],
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        # Update layout
        fig.update_layout(
            title='3D Structure',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            width=600,
            height=600
        )
        
        return fig
        
    def _create_descriptor_plot(self):
        """Create descriptor visualization."""
        if not self.properties:
            self.calculate_properties()
            
        # Create figure with numerical properties
        numerical_props = {k: v for k, v in self.properties.items() 
                         if isinstance(v, (int, float))}
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(numerical_props.keys()),
                y=list(numerical_props.values()),
                text=list(numerical_props.values()),
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title='Molecular Descriptors',
            xaxis_title='Property',
            yaxis_title='Value',
            width=800,
            height=400,
            xaxis_tickangle=-45
        )
        
        return fig

def run_chemml_analysis(analysis_type='visualization', **kwargs):
    """Run chemical visualization and analysis."""
    try:
        print(f"Starting chemical visualization with type: {analysis_type}")
        
        # Initialize visualizer
        visualizer = MoleculeVisualizer()
        
        if analysis_type == 'visualization':
            # Load molecule
            input_data = kwargs.get('input_data', '')
            input_type = kwargs.get('input_type', 'smiles')
            viz_type = kwargs.get('viz_type', '2d')
            
            if not visualizer.load_molecule(input_data, input_type):
                raise ValueError("Failed to load molecule")
                
            # Calculate properties
            properties = visualizer.calculate_properties()
            
            # Create visualization
            fig = visualizer.create_visualization(viz_type)
            
            if fig is None:
                raise ValueError("Failed to create visualization")
                
            # Return both figure and properties
            return {
                'plot': fig,
                'properties': properties
            }
            
    except Exception as e:
        print(f"Error in chemical visualization: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return {'plot': fig, 'properties': None} 