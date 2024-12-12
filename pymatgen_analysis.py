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
from pymatgen.electronic_structure.plotter import BSPlotter
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.io.vasp.inputs import Kpoints

class MaterialAnalyzer:
    def __init__(self):
        self.structure = None
        self.slab = None
        self.defect_structure = None
        
    def _create_error_figure(self, error_message):
        """Helper method to create an error figure."""
        fig = go.Figure()
        fig.add_annotation(
            text=error_message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(
                size=14,
                color="red"
            )
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False)
        )
        return fig
        
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
        pass
        
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
        
    def calculate_bandstructure(self, kpoint_density=20, energy_range=10, include_spin=False):
        """Calculate electronic band structure."""
        try:
            # Get high symmetry k-points
            kpath = HighSymmKpath(self.structure)
            kpoints = Kpoints.automatic_linemode(
                divisions=kpoint_density,
                ibz=kpath
            )
            
            # Generate dummy band structure data for demonstration
            # In a real application, you would use a DFT calculator here
            num_bands = 10
            num_kpoints = len(kpoints.kpts)
            
            # Create energy bands (dummy data)
            energies = []
            for i in range(num_bands):
                base_energy = -5 + i * 2  # Spread bands from -5 to 15 eV
                band = []
                for k in range(num_kpoints):
                    # Add some dispersion to make it look like a band structure
                    energy = base_energy + 0.5 * np.sin(k * np.pi / num_kpoints)
                    band.append(energy)
                energies.append(band)
            
            return {
                'energies': np.array(energies),
                'kpoints': kpoints.kpts,
                'labels': kpath.kpath['path'],
                'efermi': 0.0  # Fermi energy
            }
        except Exception as e:
            print(f"Error calculating band structure: {str(e)}")
            return None
    
    def create_visualization(self, plot_type='bandstructure', **kwargs):
        """Create visualization of material analysis."""
        if plot_type == 'structure':
            # Create structure visualization
            fig = go.Figure()
            
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
                )
            )
            
            # Update layout for structure
            fig.update_layout(
                title='Crystal Structure',
                scene=dict(
                    xaxis_title='X (Å)',
                    yaxis_title='Y (Å)',
                    zaxis_title='Z (Å)',
                    aspectmode='cube'
                ),
                height=600,
                showlegend=True
            )
            
            return fig
        
        elif plot_type == 'bandstructure':
            # Calculate band structure
            bs_data = self.calculate_bandstructure(
                kpoint_density=kwargs.get('kpoint_density', 20),
                energy_range=kwargs.get('energy_range', 10),
                include_spin=kwargs.get('include_spin', False)
            )
            
            if bs_data is None:
                return self._create_error_figure("Failed to calculate band structure")
            
            # Create band structure plot
            fig = go.Figure()
            
            # Plot each band
            for band in bs_data['energies']:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(band))),
                        y=band,
                        mode='lines',
                        line=dict(color='blue'),
                        showlegend=False
                    )
                )
            
            # Update layout for band structure
            fig.update_layout(
                title='Electronic Band Structure',
                xaxis_title='k-points',
                yaxis_title='Energy (eV)',
                yaxis_zeroline=True,
                yaxis=dict(
                    range=[-kwargs.get('energy_range', 10)/2, 
                           kwargs.get('energy_range', 10)/2]
                ),
                showlegend=False,
                height=600
            )
            
            # Add Fermi level line
            fig.add_hline(y=bs_data['efermi'], line_dash="dash", 
                         line_color="red", annotation_text="E_F")
            
            return fig

def run_pymatgen_analysis(material_type='bulk', analysis_type='bandstructure', **kwargs):
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
        fig = go.Figure()
        fig.add_annotation(
            text=f"Analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 