"""
ml.rl.dqn
~~~~~~~~~
Phase 17: Simple Numpy-based DQN baseline to evaluate offline RL.
Uses a small MLP (128 -> 128 -> 64 -> 5).
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List


class NumpyMLP:
    """A minimal, self-contained Multi-Layer Perceptron using NumPy."""
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights
        for i in range(len(layer_sizes) - 1):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            # He initialization
            w = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
            b = np.zeros(out_dim)
            self.weights.append(w)
            self.biases.append(b)
            
    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
        
    def _relu_deriv(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(float)
        
    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        """Returns activations at each layer."""
        activations = [X]
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self._relu(z)
            activations.append(a)
            
        # Final layer (linear, no ReLU)
        z_final = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        activations.append(z_final)
        
        return activations
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[-1]
        
    def train_step(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray = None):
        """
        Backpropagation.
        y: Target Q-values.
        mask: Optional mask (1 for actions we want to update, 0 otherwise).
        """
        activations = self.forward(X)
        output = activations[-1]
        
        # MSE gradient
        delta = output - y
        if mask is not None:
            delta *= mask
            
        # Backprop
        d_weights = []
        d_biases = []
        
        for i in reversed(range(len(self.weights))):
            a_prev = activations[i]
            
            dw = np.dot(a_prev.T, delta) / len(X)
            db = np.sum(delta, axis=0) / len(X)
            
            d_weights.insert(0, dw)
            d_biases.insert(0, db)
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._relu_deriv(np.dot(a_prev, self.weights[i-1]) + self.biases[i-1] if i > 1 else np.dot(X, self.weights[0]) + self.biases[0])
                
        # Gradient descent
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]


class DQNModel:
    """
    Offline DQN wrapper.
    """
    def __init__(self, input_dim: int, action_names: List[str] = ['hit', 'stand', 'double', 'split', 'surrender']):
        self.action_names = action_names
        self.action_dim = len(action_names)
        self.network = NumpyMLP([input_dim, 128, 128, 64, self.action_dim], learning_rate=0.005)
        
    def fit(self, X: np.ndarray, y: np.ndarray, action_indices: np.ndarray, epochs: int = 10, batch_size: int = 64):
        """
        X: features
        y: target Q-values for the taken actions
        action_indices: which action was taken
        """
        n_samples = len(X)
        
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_idx = indices[start_idx:start_idx+batch_size]
                
                X_batch = X[batch_idx]
                y_batch = y[batch_idx]
                a_batch = action_indices[batch_idx]
                
                # Get current predictions
                preds = self.network.predict(X_batch)
                
                # We only want to update the Q-value for the action that was taken
                target_q = preds.copy()
                mask = np.zeros_like(preds)
                
                for i in range(len(batch_idx)):
                    target_q[i, a_batch[i]] = y_batch[i]
                    mask[i, a_batch[i]] = 1.0
                    
                self.network.train_step(X_batch, target_q, mask=mask)
                
    def get_q_values(self, X: np.ndarray) -> np.ndarray:
        return self.network.predict(X)
        
    def predict(self, X: np.ndarray, legal_action_masks: np.ndarray) -> np.ndarray:
        """
        Returns best action indices, strictly masking illegal actions.
        legal_action_masks: 1 for legal, 0 for illegal.
        """
        q_vals = self.get_q_values(X)
        
        # Apply mask by setting illegal actions to negative infinity
        masked_q = np.where(legal_action_masks == 1, q_vals, -np.inf)
        return np.argmax(masked_q, axis=1)
