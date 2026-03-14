"""
Probabilistic Belief Representation
Represents uncertainty in state estimation with probability distributions.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)

@dataclass
class GaussianDistribution:
    """Multivariate Gaussian distribution."""
    mean: np.ndarray
    covariance: np.ndarray
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the distribution."""
        return np.random.multivariate_normal(self.mean, self.covariance, n_samples)
    
    def log_probability(self, x: np.ndarray) -> float:
        """Log probability of observation."""
        k = len(self.mean)
        diff = x - self.mean
        precision = np.linalg.inv(self.covariance)
        
        log_det = np.linalg.slogdet(self.covariance)[1]
        exponent = -0.5 * diff.T @ precision @ diff
        
        return -0.5 * (k * np.log(2 * np.pi) + log_det) + exponent
    
    @property
    def uncertainty(self) -> float:
        """Total uncertainty (trace of covariance)."""
        return float(np.trace(self.covariance))

@dataclass
class Particle:
    """Particle for particle filter representation."""
    state: Dict[str, Any]
    weight: float
    parent: Optional['Particle'] = None

class BeliefState:
    """
    Probabilistic belief about system state.
    
    Supports multiple representation methods:
    1. Gaussian (Kalman filter)
    2. Particle (particle filter)
    3. Histogram (grid-based)
    """
    
    def __init__(self, robot_id: str, representation: str = 'gaussian'):
        self.robot_id = robot_id
        self.representation = representation
        self.timestamp = 0.0
        self.confidence = 1.0
        
        # Gaussian representation
        self.gaussian_dist: Optional[GaussianDistribution] = None
        
        # Particle representation
        self.particles: List[Particle] = []
        self.n_particles = 1000
        
        # Histogram representation
        self.histogram: Dict[Tuple, float] = {}
        
        # Belief history
        self.history: List[Dict] = []
        self.max_history = 100
        
        logger.info(f"Belief state initialized for {robot_id} with {representation} representation")
    
    def update_gaussian(self, mean: np.ndarray, covariance: np.ndarray):
        """Update Gaussian belief."""
        self.gaussian_dist = GaussianDistribution(mean=mean, covariance=covariance)
        self.representation = 'gaussian'
        self._update_history()
    
    def update_particles(self, particles: List[Particle]):
        """Update particle belief."""
        self.particles = particles
        self.representation = 'particle'
        self._update_history()
    
    def predict(self, motion_model: callable, control_input: Dict):
        """
        Predict belief forward using motion model.
        
        Args:
            motion_model: Function that predicts next state
            control_input: Control inputs for prediction
        """
        if self.representation == 'gaussian' and self.gaussian_dist:
            # Extended Kalman Filter prediction
            # Simplified - in practice would use proper EKF equations
            mean = self.gaussian_dist.mean
            cov = self.gaussian_dist.covariance
            
            # Process noise
            Q = np.eye(len(mean)) * 0.01
            
            # Predict mean (simplified)
            predicted_mean = mean + np.array([0.1, 0.1, 0.0])  # Simplified
            
            # Predict covariance
            predicted_cov = cov + Q
            
            self.gaussian_dist = GaussianDistribution(
                mean=predicted_mean,
                covariance=predicted_cov
            )
            
        elif self.representation == 'particle':
            # Particle filter prediction
            for particle in self.particles:
                # Apply motion model to each particle
                particle.state = motion_model(particle.state, control_input)
        
        self.confidence *= 0.95  # Slight confidence decay on prediction
        self._update_history()
    
    def update(self, observation: Dict, observation_model: callable):
        """
        Update belief with new observation.
        
        Args:
            observation: New observation data
            observation_model: Function that computes observation likelihood
        """
        if self.representation == 'gaussian' and self.gaussian_dist:
            # Kalman filter update (simplified)
            mean = self.gaussian_dist.mean
            cov = self.gaussian_dist.covariance
            
            # Measurement noise
            R = np.eye(len(observation.get('data', []))) * 0.1
            
            # Simplified update
            # In practice: K = P * H^T * (H * P * H^T + R)^-1
            #              x = x + K * (z - H * x)
            #              P = (I - K * H) * P
            
            # For simplicity, just reduce covariance
            updated_cov = cov * 0.8
            
            self.gaussian_dist = GaussianDistribution(
                mean=mean,
                covariance=updated_cov
            )
            
            self.confidence = min(1.0, self.confidence * 1.1)
            
        elif self.representation == 'particle':
            # Particle filter update
            total_weight = 0.0
            
            for particle in self.particles:
                # Compute likelihood
                likelihood = observation_model(particle.state, observation)
                particle.weight *= likelihood
                total_weight += particle.weight
            
            # Normalize weights
            if total_weight > 0:
                for particle in self.particles:
                    particle.weight /= total_weight
            
            # Effective sample size
            ess = 1.0 / sum(w**2 for w in self.particles)
            
            if ess < len(self.particles) * 0.5:
                self._resample_particles()
            
            self.confidence = min(1.0, self.confidence * 1.05)
        
        self._update_history()
    
    def _resample_particles(self):
        """Resample particles based on weights."""
        if not self.particles:
            return
        
        weights = [p.weight for p in self.particles]
        total_weight = sum(weights)
        
        if total_weight <= 0:
            # Reset weights
            for particle in self.particles:
                particle.weight = 1.0 / len(self.particles)
            total_weight = 1.0
        
        # Systematic resampling
        indices = []
        cumulative_sum = np.cumsum(weights) / total_weight
        step = 1.0 / len(self.particles)
        position = np.random.random() * step
        
        for i in range(len(self.particles)):
            while position > cumulative_sum[indices[-1] if indices else 0]:
                indices.append(len(indices))
            position += step
        
        # Create new particles
        new_particles = []
        for idx in indices:
            if idx < len(self.particles):
                new_particle = Particle(
                    state=self.particles[idx].state.copy(),
                    weight=1.0 / len(self.particles),
                    parent=self.particles[idx]
                )
                new_particles.append(new_particle)
        
        self.particles = new_particles
    
    def get_most_likely_state(self) -> Dict[str, Any]:
        """Get the most likely state from belief."""
        if self.representation == 'gaussian' and self.gaussian_dist:
            return {
                'mean': self.gaussian_dist.mean.tolist(),
                'covariance': self.gaussian_dist.covariance.tolist(),
                'uncertainty': self.gaussian_dist.uncertainty,
                'representation': 'gaussian'
            }
        
        elif self.representation == 'particle' and self.particles:
            # Find particle with highest weight
            best_particle = max(self.particles, key=lambda p: p.weight)
            return {
                'state': best_particle.state,
                'weight': best_particle.weight,
                'representation': 'particle',
                'effective_samples': len(self.particles)
            }
        
        else:
            return {
                'error': 'No belief representation available',
                'representation': self.representation,
                'confidence': self.confidence
            }
    
    def get_uncertainty(self) -> float:
        """Get total uncertainty of belief."""
        if self.representation == 'gaussian' and self.gaussian_dist:
            return self.gaussian_dist.uncertainty
        
        elif self.representation == 'particle' and self.particles:
            # Variance of particle weights
            weights = [p.weight for p in self.particles]
            if weights:
                return float(np.var(weights))
        
        return 1.0  # Maximum uncertainty
    
    def _update_history(self):
        """Update belief history."""
        history_entry = {
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'uncertainty': self.get_uncertainty(),
            'representation': self.representation
        }
        
        self.history.append(history_entry)
        if len(self.history) > self.max_history:
            self.history.pop(0)
    
    def get_belief_statistics(self) -> Dict[str, Any]:
        """Get statistics about belief state."""
        return {
            'robot_id': self.robot_id,
            'timestamp': self.timestamp,
            'confidence': self.confidence,
            'uncertainty': self.get_uncertainty(),
            'representation': self.representation,
            'history_size': len(self.history),
            'n_particles': len(self.particles) if self.representation == 'particle' else 0
        }
    
    def reset(self):
        """Reset belief state."""
        self.gaussian_dist = None
        self.particles = []
        self.histogram = {}
        self.history = []
        self.confidence = 1.0
        logger.info("Belief state reset")
        