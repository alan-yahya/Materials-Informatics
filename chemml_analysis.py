import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

class ChemDataAnalyzer:
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_dataset(self, dataset_type='synthetic'):
        """Load or create example dataset."""
        if dataset_type == 'synthetic':
            # Create synthetic chemical data
            np.random.seed(42)
            n_samples = 1000
            
            # Features: molecular weight, number of atoms, temperature
            X = np.random.rand(n_samples, 3)
            X[:, 0] *= 200  # molecular weight (0-200)
            X[:, 1] = np.round(X[:, 1] * 20)  # number of atoms (0-20)
            X[:, 2] *= 500  # temperature (0-500)
            
            # Target: some property (e.g., solubility)
            y = 2.0 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2] + np.random.normal(0, 10, n_samples)
            
            self.data = {
                'X': X,
                'y': y,
                'feature_names': ['Molecular Weight', 'Number of Atoms', 'Temperature'],
                'target_name': 'Solubility'
            }
            return True
        return False
        
    def prepare_data(self, test_size=0.2, random_state=42):
        """Prepare data for modeling."""
        if self.data is None:
            return False
            
        X = self.data['X']
        y = self.data['y']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        self.results['train'] = {
            'X': X_train,
            'y': y_train
        }
        self.results['test'] = {
            'X': X_test,
            'y': y_test
        }
        
        return True
        
    def train_model(self, model_type='mlp', **kwargs):
        """Train a machine learning model."""
        if 'train' not in self.results:
            return False
            
        X_train = self.results['train']['X']
        y_train = self.results['train']['y']
        
        if model_type == 'mlp':
            # Create MLP model
            hidden_layers = kwargs.get('nhidden', (100, 50))
            if isinstance(hidden_layers, str):
                hidden_layers = tuple(map(int, hidden_layers.split(',')))
                
            self.model = MLPRegressor(
                hidden_layer_sizes=hidden_layers,
                activation=kwargs.get('activation', 'relu'),
                learning_rate_init=kwargs.get('learning_rate', 0.001),
                max_iter=1000,
                random_state=42
            )
            
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(self.results['test']['X'])
        
        # Calculate R² scores
        from sklearn.metrics import r2_score
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(self.results['test']['y'], y_test_pred)
        
        # Store predictions and scores
        self.results['train']['y_pred'] = y_train_pred
        self.results['test']['y_pred'] = y_test_pred
        self.results['scores'] = {
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return True
        
    def analyze_features(self):
        """Analyze feature importance."""
        if self.model is None or self.data is None:
            return None
            
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            # For MLP, use input layer weights
            importance = np.abs(self.model.coefs_[0]).mean(axis=1)
            
        return dict(zip(self.data['feature_names'], importance))
        
    def create_visualization(self, plot_type='training'):
        """Create visualization of analysis results."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=('Training Results', 'Test Results'))
        
        if plot_type == 'training' and 'train' in self.results:
            # Plot training results
            fig.add_trace(
                go.Scatter(
                    x=self.results['train']['y'],
                    y=self.results['train']['y_pred'],
                    mode='markers',
                    name='Training',
                    marker=dict(color='blue')
                ),
                row=1, col=1
            )
            
            # Add perfect prediction line
            y_min = min(self.results['train']['y'])
            y_max = max(self.results['train']['y'])
            fig.add_trace(
                go.Scatter(
                    x=[y_min, y_max],
                    y=[y_min, y_max],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=1
            )
            
            # Plot test results
            fig.add_trace(
                go.Scatter(
                    x=self.results['test']['y'],
                    y=self.results['test']['y_pred'],
                    mode='markers',
                    name='Test',
                    marker=dict(color='green')
                ),
                row=1, col=2
            )
            
            # Add perfect prediction line for test
            y_min = min(self.results['test']['y'])
            y_max = max(self.results['test']['y'])
            fig.add_trace(
                go.Scatter(
                    x=[y_min, y_max],
                    y=[y_min, y_max],
                    mode='lines',
                    name='Perfect Prediction',
                    line=dict(color='red', dash='dash')
                ),
                row=1, col=2
            )
            
            # Add R² scores to titles
            if 'scores' in self.results:
                fig.update_layout(
                    title=f'Model Performance (Train R² = {self.results["scores"]["train_r2"]:.3f}, Test R² = {self.results["scores"]["test_r2"]:.3f})'
                )
            
        # Update layout
        fig.update_layout(
            xaxis_title='Actual Values',
            yaxis_title='Predicted Values',
            xaxis2_title='Actual Values',
            yaxis2_title='Predicted Values',
            height=600,
            showlegend=True
        )
        
        # Add feature importance if available
        importance = self.analyze_features()
        if importance:
            fig.descriptors = importance
        
        return fig

def run_chemml_analysis(analysis_type='training', **kwargs):
    """Run chemical data analysis."""
    try:
        print(f"Starting chemical data analysis with type: {analysis_type}")
        
        # Initialize analyzer
        analyzer = ChemDataAnalyzer()
        
        if analysis_type == 'training':
            # Load dataset
            analyzer.load_dataset(dataset_type=kwargs.get('dataset_type', 'synthetic'))
            print("Dataset loaded")
            
            # Prepare data
            analyzer.prepare_data(
                test_size=kwargs.get('test_size', 0.2),
                random_state=kwargs.get('random_state', 42)
            )
            print("Data prepared")
            
            # Train model
            analyzer.train_model(
                model_type=kwargs.get('model_type', 'mlp'),
                nhidden=kwargs.get('nhidden', (100, 50)),
                activation=kwargs.get('activation', 'relu'),
                learning_rate=kwargs.get('learning_rate', 0.001)
            )
            print("Model trained")
            
        # Create visualization
        fig = analyzer.create_visualization(analysis_type)
        print("Visualization created")
        
        return fig
        
    except Exception as e:
        print(f"Error in chemical data analysis: {str(e)}")
        # Create error figure
        fig = go.Figure()
        fig.add_annotation(
            text=f"Analysis failed: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        return fig 