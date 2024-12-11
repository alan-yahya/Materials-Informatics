import numpy as np
import psi4
import plotly.graph_objects as go
from skimage import measure

class QuantumAtomSimulation:
    def __init__(self, atom_symbol='H', basis_set='sto-3g'):
        """
        Initialize quantum chemistry simulation for a specific atom.
        
        Parameters:
        -----------
        atom_symbol : str, optional
            Chemical symbol of the atom (default is 'H')
        basis_set : str, optional
            Basis set for quantum chemistry calculations (default is 'sto-3g')
        """
        self.atomic_numbers = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 
            'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10
        }
        
        # Atom and calculation parameters
        self.atom_symbol = atom_symbol
        self.basis_set = basis_set
        self.Z = self.atomic_numbers.get(atom_symbol, 1)
        
        # Psi4 configuration
        psi4.set_memory('2 GB')
        psi4.core.set_output_file('psi4_output.dat', False)
        
        # Prepare molecule geometry
        self._prepare_molecule()
    
    def _prepare_molecule(self):
        """
        Prepare Psi4 molecule geometry based on atom symbol.
        Uses simple linear geometry for diatomic-like representation.
        """
        # Simple geometry generation (can be expanded for more complex atoms)
        if self.atom_symbol == 'H':
            geometry = f"{self.atom_symbol} 0 0 0"
        else:
            # More complex atoms might need different initial geometries
            geometry = f"{self.atom_symbol} 0 0 0\n{self.atom_symbol} 0 0 2.0"
        
        self.molecule = psi4.geometry(geometry)
        self.molecule.set_basis_all_atoms(self.basis_set)
    
    def run_quantum_calculation(self, method='scf'):
        """
        Run quantum chemistry calculation.
        
        Parameters:
        -----------
        method : str, optional
            Quantum chemistry method (default is 'scf')
        
        Returns:
        --------
        dict: Calculation results
        """
        # Perform energy calculation
        psi4.set_options({'basis': self.basis_set})
        
        try:
            energy = psi4.energy(method)
            wavefunction = psi4.core.Wavefunction.build(self.molecule, psi4.core.get_global_option('BASIS'))
            
            return {
                'energy': energy,
                'electronic_density': self._extract_electron_density(wavefunction)
            }
        except Exception as e:
            print(f"Calculation error: {e}")
            return None
    
    def _extract_electron_density(self, wavefunction, grid_points=50, radius=5.0):
        """
        Extract and map electron density from Psi4 wavefunction.
        
        Parameters:
        -----------
        wavefunction : psi4.core.Wavefunction
            Quantum chemistry wavefunction
        grid_points : int, optional
            Number of grid points (default 50)
        radius : float, optional
            Simulation radius (default 5.0 Angstroms)
        
        Returns:
        --------
        tuple: Coordinate grids and electron density
        """
        # Create 3D grid
        x = np.linspace(-radius, radius, grid_points)
        y = np.linspace(-radius, radius, grid_points)
        z = np.linspace(-radius, radius, grid_points)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Placeholder for more advanced density calculation
        # This is a simplified approximation
        r = np.sqrt(X**2 + Y**2 + Z**2)
        density = np.exp(-2 * r / self.Z)  # Basic radial probability distribution
        
        return x, y, z, density
    
    def visualize_electron_density(self, density_data, isovalue=0.01):
        """
        Create 3D visualization of electron density using Plotly.
        
        Parameters:
        -----------
        density_data : tuple
            Coordinate grids and density from _extract_electron_density
        isovalue : float, optional
            Isosurface threshold value
        
        Returns:
        --------
        plotly.graph_objects.Figure
        """
        x, y, z, density = density_data
        
        # Create isosurface
        vertices, faces = self._create_isosurface(x, y, z, density, isovalue)
        
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.5,
                colorscale='Viridis',
                intensity=np.linalg.norm(vertices, axis=1),
            )
        ])
        
        fig.update_layout(
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube'
            ),
            title=f'Electron Density for {self.atom_symbol} Atom'
        )
        
        return fig
    
    def _create_isosurface(self, x, y, z, density, isovalue):
        """
        Create vertices and faces for isosurface visualization.
        
        Parameters:
        -----------
        x, y, z : numpy.ndarray
            Coordinate grids
        density : numpy.ndarray
            Electron density grid
        isovalue : float
            Isosurface threshold
        
        Returns:
        --------
        tuple: Vertices and faces for 3D visualization
        """
        verts, faces, _, _ = measure.marching_cubes(density, isovalue)
        
        # Scale vertices to match coordinate system
        verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (density.shape[0] - 1)
        verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (density.shape[1] - 1)
        verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (density.shape[2] - 1)
        
        return verts, faces 