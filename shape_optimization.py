import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import splprep, splev
import plotly.graph_objects as go

class ShapeOptimizer:
    def __init__(self, n_points=50, alpha=0.1):
        self.n_points = n_points
        self.alpha = alpha  # Smoothing parameter
        
    def initialize_shape(self):
        # Create initial circular shape
        theta = np.linspace(0, 2*np.pi, self.n_points)
        x = np.cos(theta)
        y = np.sin(theta)
        return np.column_stack((x, y))
        
    def objective_function(self, points):
        # Compute area and perimeter
        x = points[:, 0]
        y = points[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        dx = np.diff(x, append=x[0])
        dy = np.diff(y, append=y[0])
        perimeter = np.sum(np.sqrt(dx**2 + dy**2))
        
        # Penalize non-smooth shapes
        smoothness = np.sum(np.diff(dx)**2 + np.diff(dy)**2)
        
        return perimeter + self.alpha * smoothness - area

    def optimize(self, max_iter=100):
        initial_shape = self.initialize_shape()
        shape_flat = initial_shape.flatten()
        
        # Constraint to maintain closed shape
        constraints = [{
            'type': 'eq',
            'fun': lambda x: x[0] - x[-2],
        }, {
            'type': 'eq',
            'fun': lambda x: x[1] - x[-1],
        }]
        
        result = minimize(
            lambda x: self.objective_function(x.reshape(-1, 2)),
            shape_flat,
            method='SLSQP',
            constraints=constraints,
            options={'maxiter': max_iter}
        )
        
        return result.x.reshape(-1, 2), result.fun

def run_shape_optimization(n_points=50, max_iter=100):
    optimizer = ShapeOptimizer(n_points=n_points)
    initial_shape = optimizer.initialize_shape()
    optimized_shape, obj_value = optimizer.optimize(max_iter=max_iter)
    
    # Create visualization
    fig = go.Figure()
    
    # Plot initial shape
    fig.add_trace(go.Scatter(
        x=initial_shape[:, 0],
        y=initial_shape[:, 1],
        mode='lines',
        name='Initial Shape',
        line=dict(dash='dash', color='blue')
    ))
    
    # Plot optimized shape
    fig.add_trace(go.Scatter(
        x=optimized_shape[:, 0],
        y=optimized_shape[:, 1],
        mode='lines',
        name='Optimized Shape',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Nanoparticle Shape Optimization',
        xaxis_title='X',
        yaxis_title='Y',
        showlegend=True,
        width=600,
        height=600
    )
    
    return fig 