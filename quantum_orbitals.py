import numpy as np
from scipy.special import sph_harm, genlaguerre
import plotly.graph_objects as go
from skimage import measure

def hydrogen_wavefunction(n, l, m, r, theta, phi):
    """Calculate hydrogen-like atomic orbital wavefunction."""
    # Normalization constant
    rho = 2.0 * r / n
    # Radial part (using associated Laguerre polynomials)
    L = genlaguerre(n-l-1, 2*l+1)(rho)
    R = np.sqrt((2.0/n)**3 * np.math.factorial(n-l-1)/(2*n*np.math.factorial(n+l))) * \
        np.exp(-rho/2) * rho**l * L
    # Angular part (using spherical harmonics)
    Y = sph_harm(m, l, phi, theta)
    return R * Y

def create_atom(atom_symbol='H', basis='basic'):
    """Store atomic parameters."""
    atomic_numbers = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 
                     'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10}
    
    return {
        'symbol': atom_symbol,
        'Z': atomic_numbers.get(atom_symbol, 1),
        'basis': basis
    }

def calculate_orbital_wavefunction(atom_params, grid_points=50, radius=5.0):
    """Calculate orbital wavefunction on a 3D grid."""
    # Create 3D grid
    x = np.linspace(-radius, radius, grid_points)
    y = np.linspace(-radius, radius, grid_points)
    z = np.linspace(-radius, radius, grid_points)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Convert to spherical coordinates
    r = np.sqrt(X**2 + Y**2 + Z**2)
    theta = np.arccos(Z/np.where(r == 0, 1, r))
    phi = np.arctan2(Y, X)
    
    # Initialize density array
    density = np.zeros_like(r)
    
    # Calculate electron density based on atomic number
    Z = atom_params['Z']
    
    # Add contributions from occupied orbitals
    if Z >= 1:  # 1s orbital
        psi_1s = hydrogen_wavefunction(1, 0, 0, Z*r, theta, phi)
        density += 2 * np.abs(psi_1s)**2  # Factor of 2 for spin degeneracy
        
    if Z >= 3:  # 2s orbital
        psi_2s = hydrogen_wavefunction(2, 0, 0, Z*r, theta, phi)
        density += 2 * np.abs(psi_2s)**2
        
    if Z >= 5:  # 2p orbitals
        for m in [-1, 0, 1]:
            psi_2p = hydrogen_wavefunction(2, 1, m, Z*r, theta, phi)
            density += 2 * np.abs(psi_2p)**2
    
    return x, y, z, np.real(density)

def plot_orbital_density(x, y, z, density, isovalue=0.01):
    """Create a 3D visualization of orbital density using Plotly."""
    # Create vertices and faces for isosurface
    vertices, faces = create_isosurface(x, y, z, density, isovalue)
    
    # Create mesh3d trace
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
        title='Atomic Orbital Electron Density'
    )
    
    return fig

def create_isosurface(x, y, z, density, isovalue):
    """Create vertices and faces for isosurface visualization."""
    # Generate isosurface vertices and faces
    verts, faces, _, _ = measure.marching_cubes(density, isovalue)
    
    # Scale vertices to match coordinate system
    verts[:, 0] = x[0] + (x[-1] - x[0]) * verts[:, 0] / (density.shape[0] - 1)
    verts[:, 1] = y[0] + (y[-1] - y[0]) * verts[:, 1] / (density.shape[1] - 1)
    verts[:, 2] = z[0] + (z[-1] - z[0]) * verts[:, 2] / (density.shape[2] - 1)
    
    return verts, faces

def get_available_atoms():
    """Return a list of available atoms for simulation."""
    return ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne']

def get_available_basis_sets():
    """Return a list of available basis sets."""
    return ['basic', 'extended']  # Simplified basis set options