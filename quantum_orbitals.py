import numpy as np
from scipy.special import sph_harm, genlaguerre, factorial
import plotly.graph_objects as go
from skimage import measure

def hydrogen_wavefunction(n, l, m, r, theta, phi):
    """
    Calculate hydrogen-like atomic orbital wavefunction.
    
    Parameters:
        n (int): Principal quantum number
        l (int): Angular momentum quantum number 
        m (int): Magnetic quantum number
        r (ndarray): Radial coordinates
        theta (ndarray): Polar angle coordinates
        phi (ndarray): Azimuthal angle coordinates
        
    Returns:
        ndarray: Complex wavefunction values
    """
    # Input validation
    if n < 1:
        raise ValueError("Principal quantum number n must be >= 1")
    if l < 0 or l >= n:
        raise ValueError(f"Angular momentum l must be 0 <= l < n (got l={l}, n={n})")
    if abs(m) > l:
        raise ValueError(f"Magnetic quantum number |m| must be <= l (got m={m}, l={l})")

    # Bohr radius in Angstroms
    a0 = 0.529177210903

    # Normalized radius
    rho = 2.0 * r / (n * a0)
    
    # Normalization constant including all terms
    norm = np.sqrt((2.0/(n*a0))**3 * factorial(n-l-1)/(2*n*factorial(n+l)))
    
    # Radial part using associated Laguerre polynomials
    L = genlaguerre(n-l-1, 2*l+1)(rho)
    R = norm * np.exp(-rho/2) * rho**l * L
    
    # Angular part using spherical harmonics
    Y = sph_harm(m, l, phi, theta)
    
    return R * Y

def create_atom(atom_symbol='H', basis='basic'):
    """
    Store atomic parameters and electron configuration.
    
    Parameters:
        atom_symbol (str): Chemical symbol of the atom
        basis (str): Basis set name
        
    Returns:
        dict: Atomic parameters including electron configuration
    """
    # Electron configurations for first 10 elements
    configurations = {
        'H':  {'1s': 1},
        'He': {'1s': 2},
        'Li': {'1s': 2, '2s': 1},
        'Be': {'1s': 2, '2s': 2},
        'B':  {'1s': 2, '2s': 2, '2p': 1},
        'C':  {'1s': 2, '2s': 2, '2p': 2},
        'N':  {'1s': 2, '2s': 2, '2p': 3},
        'O':  {'1s': 2, '2s': 2, '2p': 4},
        'F':  {'1s': 2, '2s': 2, '2p': 5},
        'Ne': {'1s': 2, '2s': 2, '2p': 6}
    }
    
    atomic_numbers = {symbol: i+1 for i, symbol in enumerate(configurations.keys())}
    
    if atom_symbol not in atomic_numbers:
        raise ValueError(f"Unsupported atom: {atom_symbol}")
        
    return {
        'symbol': atom_symbol,
        'Z': atomic_numbers[atom_symbol],
        'configuration': configurations[atom_symbol],
        'basis': basis
    }

def calculate_orbital_wavefunction(atom_params, grid_points=200, radius=8.0, adaptive=True):
    """
    Calculate orbital wavefunction on a 3D grid.
    
    Parameters:
        atom_params (dict): Atomic parameters from create_atom()
        grid_points (int): Number of points along each axis (50-500 recommended)
        radius (float): Maximum radius in Angstroms
        adaptive (bool): Use adaptive grid spacing for better near-nucleus resolution
        
    Returns:
        tuple: (x, y, z coordinates and electron density)
    """
    # Validate grid points
    if not isinstance(grid_points, int):
        raise TypeError("grid_points must be an integer")
    if grid_points < 50:
        raise ValueError("grid_points must be at least 50 for meaningful results")
    if grid_points > 500:
        print("Warning: High grid_points value may require significant memory and computation time")
    
    if adaptive:
        # Create non-uniform grid with higher density near nucleus
        # Use hyperbolic tangent transformation for smooth transition
        beta = 2.0  # Controls grid density distribution
        uniform = np.linspace(-1, 1, grid_points)
        transform = np.tanh(beta * uniform) / np.tanh(beta)
        x = radius * transform
        y = radius * transform
        z = radius * transform
    else:
        # Uniform grid
        x = np.linspace(-radius, radius, grid_points)
        y = np.linspace(-radius, radius, grid_points)
        z = np.linspace(-radius, radius, grid_points)
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Convert to spherical coordinates with careful handling of special cases
    r = np.sqrt(X**2 + Y**2 + Z**2)
    
    # Handle points very close to origin separately to avoid numerical issues
    near_zero = r < 1e-10
    r[near_zero] = 1e-10
    
    theta = np.arccos(np.clip(Z/r, -1, 1))
    phi = np.arctan2(Y, X)
    
    # Initialize density array with small positive value to avoid numerical issues
    density = np.full_like(r, 1e-20)
    Z = atom_params['Z']
    config = atom_params['configuration']
    
    # Quantum numbers for each orbital
    orbital_params = {
        '1s': (1, 0, 0),
        '2s': (2, 0, 0),
        '2p': [(2, 1, m) for m in [-1, 0, 1]]
    }
    
    # Add contributions from each occupied orbital
    for orbital, electrons in config.items():
        if orbital == '2p':
            # Split electrons among p orbitals
            for n, l, m in orbital_params[orbital]:
                psi = hydrogen_wavefunction(n, l, m, Z*r, theta, phi)
                density += (electrons/3) * np.abs(psi)**2
        else:
            n, l, m = orbital_params[orbital]
            psi = hydrogen_wavefunction(n, l, m, Z*r, theta, phi)
            density += electrons * np.abs(psi)**2
    
    # Set density to zero at points very close to origin to avoid singularity
    density[near_zero] = 0
    
    return x, y, z, np.real(density)

def plot_orbital_density(x, y, z, density, isovalue=0.01):
    """
    Create a 3D visualization of orbital density using Plotly.
    
    Parameters:
        x, y, z (ndarray): Coordinate arrays
        density (ndarray): Electron density values
        isovalue (float): Isosurface threshold value
    
    Returns:
        plotly.graph_objects.Figure: Interactive 3D plot
    """
    # Create vertices and faces for isosurface
    vertices, faces = create_isosurface(x, y, z, density, isovalue)
    
    # Calculate vertex colors based on distance from origin
    distances = np.linalg.norm(vertices, axis=1)
    colors = np.clip(1.0 - distances/np.max(distances), 0, 1)
    
    # Create mesh3d trace
    fig = go.Figure(data=[
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=0.7,
            colorscale='Viridis',
            intensity=colors,
            showscale=True,
            name='Electron Density',
            colorbar=dict(
                title='Relative Density',
                titleside='right'
            )
        )
    ])
    
    # Add atom position marker
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers',
        marker=dict(
            size=10, 
            color='red',
            symbol='circle',
            line=dict(color='darkred', width=2)
        ),
        name='Nucleus'
    ))
    
    # Update layout with better defaults
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5),
                up=dict(x=0, y=0, z=1)
            ),
            annotations=[
                dict(
                    showarrow=False,
                    x=0, y=0, z=0,
                    text="Nucleus",
                    xanchor="left",
                    xshift=10,
                    opacity=0.7
                )
            ]
        ),
        title='Atomic Orbital Electron Density',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.8)'
        )
    )
    
    return fig

def create_isosurface(x, y, z, density, isovalue):
    """
    Create vertices and faces for isosurface visualization.
    
    Parameters:
        x, y, z (ndarray): Coordinate arrays
        density (ndarray): Electron density values
        isovalue (float): Isosurface threshold value
        
    Returns:
        tuple: (vertices, faces) for 3D mesh
    """
    try:
        verts, faces, _, _ = measure.marching_cubes(density, isovalue)
    except ValueError as e:
        print(f"Warning: Marching cubes failed with isovalue {isovalue}. Trying adjusted value.")
        # Try with adjusted isovalue if original fails
        isovalue = np.percentile(density, 95)
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
    return ['minimal', 'basic', 'extended', '6-31g']