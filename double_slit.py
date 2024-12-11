import numpy as np
import plotly.graph_objects as go
from scipy.constants import c

def calculate_interference_pattern(slit_width, slit_separation, wavelength, screen_distance=0.1, screen_width=0.1, num_points=500):
    """Calculate double-slit interference pattern using wave superposition."""
    k = 2 * np.pi / wavelength  # Wave number
    
    # Create screen coordinates
    screen_x = np.linspace(-screen_width/2, screen_width/2, num_points)
    screen_y = np.linspace(-screen_width/2, screen_width/2, num_points)
    X, Y = np.meshgrid(screen_x, screen_y)
    
    # Positions of the slits
    slit1_pos = np.array([0, -slit_separation/2, 0])
    slit2_pos = np.array([0, slit_separation/2, 0])
    
    # Calculate distances from each point on screen to slits
    R1 = np.sqrt(screen_distance**2 + (X - slit1_pos[1])**2 + Y**2)
    R2 = np.sqrt(screen_distance**2 + (X - slit2_pos[1])**2 + Y**2)
    
    # Calculate wave amplitudes (including single-slit diffraction)
    alpha1 = k * slit_width * (X - slit1_pos[1]) / np.sqrt(X**2 + screen_distance**2)
    alpha2 = k * slit_width * (X - slit2_pos[1]) / np.sqrt(X**2 + screen_distance**2)
    
    sinc1 = np.where(alpha1 == 0, 1, np.sin(alpha1) / alpha1)
    sinc2 = np.where(alpha2 == 0, 1, np.sin(alpha2) / alpha2)
    
    # Calculate interference pattern
    wave1 = sinc1 * np.exp(1j * k * R1) / R1
    wave2 = sinc2 * np.exp(1j * k * R2) / R2
    
    # Total intensity
    intensity = np.abs(wave1 + wave2)**2
    
    # Normalize intensity
    intensity = intensity / np.max(intensity)
    
    return X, Y, intensity

def run_double_slit_simulation(slit_width=1e-6, slit_separation=5e-6, wavelength=500e-9):
    """Run double-slit simulation and create visualization."""
    # Calculate interference pattern
    X, Y, intensity = calculate_interference_pattern(
        slit_width=slit_width,
        slit_separation=slit_separation,
        wavelength=wavelength
    )
    
    # Create visualization
    fig = go.Figure(data=[
        # Interference pattern
        go.Heatmap(
            x=X[0] * 1e3,  # Convert to mm
            y=Y[:, 0] * 1e3,  # Convert to mm
            z=intensity,
            colorscale='Viridis',
            colorbar=dict(title='Relative Intensity'),
        ),
    ])
    
    # Update layout
    fig.update_layout(
        title='Double-Slit Interference Pattern',
        xaxis_title='Position (mm)',
        yaxis_title='Position (mm)',
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    
    return fig 