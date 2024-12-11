import numpy as np
import pyvista as pv
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

class NanoparticleVisualizer:
    def __init__(self):
        # Configure PyVista for off-screen rendering
        pv.OFF_SCREEN = True
        self.plotter = pv.Plotter(off_screen=True)
        
    def create_sphere_nanoparticle(self, radius=10, resolution=30):
        """Create a spherical nanoparticle."""
        sphere = pv.Sphere(radius=radius, phi_resolution=resolution, theta_resolution=resolution)
        return sphere
        
    def create_rod_nanoparticle(self, length=20, radius=5, resolution=30):
        """Create a nanorod."""
        cylinder = pv.Cylinder(radius=radius, height=length, resolution=resolution)
        return cylinder
        
    def create_cube_nanoparticle(self, size=10):
        """Create a cubic nanoparticle."""
        cube = pv.Cube(x_length=size, y_length=size, z_length=size)
        return cube
        
    def create_composite_nanoparticle(self, core_radius=8, shell_thickness=2, resolution=30):
        """Create a core-shell nanoparticle."""
        core = pv.Sphere(radius=core_radius, phi_resolution=resolution, theta_resolution=resolution)
        shell = pv.Sphere(radius=core_radius + shell_thickness, phi_resolution=resolution, theta_resolution=resolution)
        return core, shell
        
    def add_surface_ligands(self, particle, n_ligands=10, ligand_length=2):
        """Add surface ligands to particle."""
        points = particle.points
        normals = particle.compute_normals().point_data['Normals']
        
        # Randomly select points for ligands
        indices = np.random.choice(len(points), n_ligands, replace=False)
        ligands = []
        
        for idx in indices:
            start = points[idx]
            direction = normals[idx]
            end = start + direction * ligand_length
            line = pv.Line(start, end)
            ligands.append(line)
            
        return ligands
        
    def apply_electrostatic_potential(self, particle, potential_function):
        """Apply electrostatic potential to particle surface."""
        points = particle.points
        potentials = np.array([potential_function(p) for p in points])
        particle.point_data['potential'] = potentials
        return particle
        
    def create_visualization(self, particle_type='sphere', **kwargs):
        """Create visualization of nanoparticle."""
        # Clear previous plots
        self.plotter.clear()
        
        # Create particle based on type
        if particle_type == 'sphere':
            particle = self.create_sphere_nanoparticle(
                radius=kwargs.get('radius', 10),
                resolution=kwargs.get('resolution', 30)
            )
        elif particle_type == 'rod':
            particle = self.create_rod_nanoparticle(
                length=kwargs.get('length', 20),
                radius=kwargs.get('radius', 5),
                resolution=kwargs.get('resolution', 30)
            )
        elif particle_type == 'cube':
            particle = self.create_cube_nanoparticle(
                size=kwargs.get('size', 10)
            )
        elif particle_type == 'core-shell':
            core, shell = self.create_composite_nanoparticle(
                core_radius=kwargs.get('core_radius', 8),
                shell_thickness=kwargs.get('shell_thickness', 2),
                resolution=kwargs.get('resolution', 30)
            )
            # Add core and shell with different colors
            self.plotter.add_mesh(core, color='gold', opacity=1.0)
            self.plotter.add_mesh(shell, color='silver', opacity=0.5)
            particle = shell  # For ligands
        else:
            raise ValueError(f"Unknown particle type: {particle_type}")
            
        # Add main particle if not core-shell
        if particle_type != 'core-shell':
            self.plotter.add_mesh(particle, color=kwargs.get('color', 'gold'))
            
        # Add surface ligands if requested
        if kwargs.get('add_ligands', False):
            ligands = self.add_surface_ligands(
                particle,
                n_ligands=kwargs.get('n_ligands', 10),
                ligand_length=kwargs.get('ligand_length', 2)
            )
            for ligand in ligands:
                self.plotter.add_mesh(ligand, color='blue', line_width=3)
                
        # Apply electrostatic potential if function provided
        if 'potential_function' in kwargs:
            particle = self.apply_electrostatic_potential(particle, kwargs['potential_function'])
            self.plotter.add_mesh(particle, scalars='potential', cmap='rainbow')
            
        # Set camera position and take screenshot
        self.plotter.camera_position = 'iso'
        self.plotter.camera.zoom(1.5)
        
        # Render and capture image
        self.plotter.show(auto_close=False)
        image = self.plotter.screenshot(return_img=True)
        
        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Create Plotly figure with image
        fig = go.Figure()
        
        fig.add_trace(go.Image(
            source=f'data:image/png;base64,{image_base64}'
        ))
        
        fig.update_layout(
            title='Nanoparticle Visualization',
            width=800,
            height=800,
            showlegend=False
        )
        
        # Remove axes
        fig.update_xaxes(showticklabels=False, showgrid=False)
        fig.update_yaxes(showticklabels=False, showgrid=False)
        
        return fig

def run_pyvista_visualization(particle_type='sphere', **kwargs):
    """Run nanoparticle visualization using PyVista."""
    try:
        print(f"Starting PyVista visualization with particle type: {particle_type}")
        
        # Initialize visualizer
        visualizer = NanoparticleVisualizer()
        
        # Create visualization
        fig = visualizer.create_visualization(particle_type, **kwargs)
        
        print("Visualization completed successfully")
        return fig
        
    except Exception as e:
        print(f"Error in PyVista visualization: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Visualization failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 