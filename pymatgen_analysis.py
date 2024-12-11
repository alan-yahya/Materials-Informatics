import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.elasticity import Strain
from pymatgen.transformations.standard_transformations import SupercellTransformation

class MaterialAnalyzer:
    def __init__(self):
        self.structure = None
        self.slab = None
        self.defect_structure = None
        
    def create_structure(self, material_type='bulk', **kwargs):
        """Create material structure based on type."""
        if material_type == 'bulk':
            # Create bulk crystal structure
            lattice = Lattice.cubic(kwargs.get('lattice_constant', 3.5))
            species = kwargs.get('species', ['Au'])
            coords = [[0, 0, 0]]
            self.structure = Structure(lattice, species, coords)
            
        elif material_type == 'surface':
            # Create surface from bulk
            if self.structure is None:
                self.create_structure('bulk', **kwargs)
            miller_index = kwargs.get('miller_index', (1, 1, 1))
            min_slab_size = kwargs.get('min_slab_size', 10.0)
            min_vacuum_size = kwargs.get('min_vacuum_size', 10.0)
            
            slab_gen = SlabGenerator(self.structure, miller_index, min_slab_size, min_vacuum_size)
            self.slab = slab_gen.get_slab()
            self.structure = self.slab
            
        elif material_type == 'nanoparticle':
            # Create nanoparticle by cutting sphere from bulk
            if self.structure is None:
                self.create_structure('bulk', **kwargs)
            radius = kwargs.get('radius', 10.0)
            
            # Create supercell
            trans = SupercellTransformation.from_scaling_factors(5, 5, 5)
            super_structure = trans.apply_transformation(self.structure)
            
            # Keep only atoms within radius
            coords = super_structure.cart_coords
            center = np.mean(coords, axis=0)
            distances = np.linalg.norm(coords - center, axis=1)
            mask = distances <= radius
            
            species = [super_structure.species[i] for i in range(len(mask)) if mask[i]]
            coords = coords[mask]
            
            self.structure = Structure(super_structure.lattice, species, coords, coords_are_cartesian=True)
            
    def analyze_symmetry(self):
        """Analyze crystal symmetry."""
        analyzer = SpacegroupAnalyzer(self.structure)
        symmetry_data = {
            'space_group': analyzer.get_space_group_symbol(),
            'point_group': analyzer.get_point_group_symbol(),
            'crystal_system': analyzer.get_crystal_system(),
            'hall_number': analyzer.get_hall()
        }
        return symmetry_data
        
    def analyze_bonding(self, method='voronoi'):
        """Analyze bonding environment."""
        if method == 'voronoi':
            nn = VoronoiNN()
        else:
            nn = CrystalNN()
            
        # Get bonding information for each site
        bonding_info = []
        for i, site in enumerate(self.structure):
            neighbors = nn.get_nn_info(self.structure, i)
            bonding_info.append({
                'site': i,
                'element': site.specie.symbol,
                'coordination': len(neighbors),
                'neighbors': [{'element': n['site'].specie.symbol, 'distance': n['distance']} for n in neighbors]
            })
        return bonding_info
        
    def create_vacancy(self, site_index=0):
        """Create a vacancy by removing a site from the structure."""
        if self.structure is None:
            return None
        
        # Create a copy of the structure
        defect_structure = self.structure.copy()
        
        # Remove the site to create a vacancy
        defect_structure.remove_sites([site_index])
        
        self.defect_structure = defect_structure
        return self.defect_structure
        
    def apply_strain(self, strain_matrix=None):
        """Apply strain to structure."""
        if strain_matrix is None:
            # Default: 1% tensile strain along x
            strain_matrix = np.eye(3)
            strain_matrix[0, 0] = 1.01
            
        strain = Strain(strain_matrix)
        self.structure.apply_strain(strain_matrix)
        
    def create_visualization(self, plot_type='structure', **kwargs):
        """Create visualization of material analysis."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Structure', 'Analysis'),
                           specs=[[{'type': 'scatter3d'}, {'type': 'scatter'}]])
        
        # Plot structure
        coords = self.structure.cart_coords
        elements = [site.specie.symbol for site in self.structure]
        
        # Create element-color mapping
        unique_elements = list(set(elements))
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange'][:len(unique_elements)]
        element_colors = dict(zip(unique_elements, colors))
        
        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode='markers',
                marker=dict(
                    size=10,
                    color=[element_colors[el] for el in elements],
                    symbol='circle'
                ),
                text=elements,
                name='Atoms'
            ),
            row=1, col=1
        )
        
        # Plot analysis based on type
        if plot_type == 'symmetry':
            symmetry_data = self.analyze_symmetry()
            text = [f"{k}: {v}" for k, v in symmetry_data.items()]
            fig.add_trace(
                go.Scatter(
                    x=[0],
                    y=[0],
                    mode='text',
                    text='\n'.join(text),
                    textposition='middle center',
                    name='Symmetry'
                ),
                row=1, col=2
            )
            
        elif plot_type == 'bonding':
            bonding_info = self.analyze_bonding()
            coord_numbers = [info['coordination'] for info in bonding_info]
            elements = [info['element'] for info in bonding_info]
            
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(coord_numbers))),
                    y=coord_numbers,
                    mode='markers',
                    text=elements,
                    name='Coordination'
                ),
                row=1, col=2
            )
            
        # Update layout
        fig.update_layout(
            title='Material Analysis',
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)'
            ),
            height=600,
            showlegend=True
        )
        
        return fig

def run_pymatgen_analysis(material_type='bulk', analysis_type='structure', **kwargs):
    """Run materials analysis using pymatgen."""
    try:
        print(f"Starting pymatgen analysis with material type: {material_type}")
        
        # Initialize analyzer
        analyzer = MaterialAnalyzer()
        
        # Create structure
        analyzer.create_structure(material_type, **kwargs)
        print("Structure created")
        
        # Create visualization
        fig = analyzer.create_visualization(analysis_type, **kwargs)
        print("Visualization created")
        
        return fig
        
    except Exception as e:
        print(f"Error in pymatgen analysis: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 