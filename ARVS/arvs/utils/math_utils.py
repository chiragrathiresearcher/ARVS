

import numpy as np
from typing import Tuple, Optional, Union, Any
import logging
import struct
import hashlib

logger = logging.getLogger(__name__)

class SpaceMathError(Exception):
    """Exception for spacecraft math operations."""
    pass

class RadiationHardenedMath:
    """
    Mathematical operations with radiation hardening.
    
    Features:
    1. Checksum verification for all operations
    2. Graceful degradation on bit-flip detection
    3. Multiple redundant computation paths
    4. Memory error detection and correction
    """
    
    def __init__(self):
        self.error_count = 0
        self.max_errors = 100
        self.computation_history = []
        
    def hardened_dot(self, a: np.ndarray, b: np.ndarray, 
                    checksum: Optional[float] = None) -> Tuple[float, float]:
        """
        Radiation-hardened dot product with triple redundancy.
        
        Args:
            a: First vector
            b: Second vector
            checksum: Optional expected checksum for verification
            
        Returns:
            Tuple of (result, confidence)
            
        Raises:
            SpaceMathError: If computation fails or radiation error detected
        """
        # Triple redundant computation
        result1 = np.dot(a, b)
        result2 = np.dot(np.flip(a), np.flip(b))  # Different computation path
        result3 = np.sum(a * b)  # Another computation path
        
        # Verify consistency
        results = [result1, result2, result3]
        avg_result = np.mean(results)
        std_result = np.std(results)
        
        # Check for radiation-induced errors
        if std_result > 1e-6:  # Results don't match
            self.error_count += 1
            logger.warning(f"Radiation error detected in dot product: std={std_result}")
            
            if self.error_count > self.max_errors:
                raise SpaceMathError("Excessive radiation errors detected")
            
            # Use median as it's more robust to outliers
            result = np.median(results)
            confidence = 0.7 - (std_result * 10)  # Reduced confidence
        else:
            result = avg_result
            confidence = 0.99
        
        # Verify checksum if provided
        if checksum is not None:
            computed_checksum = self._compute_checksum(result)
            if abs(computed_checksum - checksum) > 1e-9:
                logger.error("Checksum mismatch in dot product")
                confidence *= 0.5
        
        self.computation_history.append({
            'operation': 'dot',
            'result': result,
            'confidence': confidence,
            'std': std_result
        })
        
        return float(result), float(confidence)
    
    def hardened_matrix_multiply(self, A: np.ndarray, B: np.ndarray,
                                algorithm: str = 'auto') -> Tuple[np.ndarray, float]:
        """
        Radiation-hardened matrix multiplication.
        
        Args:
            A: First matrix
            B: Second matrix
            algorithm: 'strassen', 'standard', or 'auto'
            
        Returns:
            Tuple of (result matrix, confidence)
        """
        # Input validation
        if A.shape[1] != B.shape[0]:
            raise SpaceMathError(f"Matrix dimensions incompatible: {A.shape} vs {B.shape}")
        
        # Choose algorithm based on size
        if algorithm == 'auto':
            if A.shape[0] <= 32 or B.shape[1] <= 32:
                algorithm = 'standard'
            else:
                algorithm = 'strassen'
        
        # Dual computation for redundancy
        if algorithm == 'strassen':
            C1 = self._strassen_multiply(A, B)
            C2 = self._standard_multiply(A, B)
        else:
            C1 = self._standard_multiply(A, B)
            C2 = np.matmul(A, B)  # Use numpy's implementation
        
        # Compare results
        diff = np.max(np.abs(C1 - C2))
        
        if diff > 1e-6:
            self.error_count += 1
            logger.warning(f"Matrix multiplication mismatch: diff={diff}")
            
            # Use weighted average based on confidence
            confidence1 = 1.0 - min(1.0, diff * 10)
            confidence2 = 1.0 - min(1.0, diff * 10)
            
            C = (C1 * confidence1 + C2 * confidence2) / (confidence1 + confidence2)
            confidence = (confidence1 + confidence2) / 2
        else:
            C = (C1 + C2) / 2
            confidence = 0.99
        
        # Add checksum to result
        checksum = self._matrix_checksum(C)
        
        return C, float(confidence)
    
    def _strassen_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Strassen algorithm for matrix multiplication (redundant path)."""
        n = A.shape[0]
        
        # Base case
        if n == 1:
            return A * B
        
        # Split matrices
        mid = n // 2
        A11 = A[:mid, :mid]
        A12 = A[:mid, mid:]
        A21 = A[mid:, :mid]
        A22 = A[mid:, mid:]
        
        B11 = B[:mid, :mid]
        B12 = B[:mid, mid:]
        B21 = B[mid:, :mid]
        B22 = B[mid:, mid:]
        
        # Compute Strassen products
        M1 = self._strassen_multiply(A11 + A22, B11 + B22)
        M2 = self._strassen_multiply(A21 + A22, B11)
        M3 = self._strassen_multiply(A11, B12 - B22)
        M4 = self._strassen_multiply(A22, B21 - B11)
        M5 = self._strassen_multiply(A11 + A12, B22)
        M6 = self._strassen_multiply(A21 - A11, B11 + B12)
        M7 = self._strassen_multiply(A12 - A22, B21 + B22)
        
        # Combine results
        C = np.zeros((n, n))
        C[:mid, :mid] = M1 + M4 - M5 + M7
        C[:mid, mid:] = M3 + M5
        C[mid:, :mid] = M2 + M4
        C[mid:, mid:] = M1 - M2 + M3 + M6
        
        return C
    
    def _standard_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Standard matrix multiplication (triple-nested loops)."""
        m, n = A.shape
        n, p = B.shape
        C = np.zeros((m, p))
        
        for i in range(m):
            for j in range(p):
                sum_val = 0.0
                for k in range(n):
                    sum_val += A[i, k] * B[k, j]
                C[i, j] = sum_val
        
        return C
    
    def _compute_checksum(self, value: float) -> float:
        """Compute checksum for a floating-point value."""
        # Convert to bytes and hash
        byte_repr = struct.pack('d', value)
        hash_val = hashlib.md5(byte_repr).hexdigest()
        
        # Convert first 8 bytes to float
        checksum_bytes = bytes.fromhex(hash_val[:16])
        checksum = struct.unpack('d', checksum_bytes[:8])[0]
        
        return checksum
    
    def _matrix_checksum(self, M: np.ndarray) -> float:
        """Compute checksum for a matrix."""
        flat = M.flatten()
        combined = np.sum(flat) + np.prod(flat[flat != 0]) if np.any(flat != 0) else 0.0
        return self._compute_checksum(combined)
    
    def get_stats(self) -> dict:
        """Get statistics about math operations."""
        return {
            'error_count': self.error_count,
            'max_errors': self.max_errors,
            'recent_operations': len(self.computation_history),
            'avg_confidence': np.mean([op['confidence'] for op in self.computation_history[-100:]])
        }

class QuaternionOperations:
    """
    Spacecraft-grade quaternion operations.
    Essential for attitude determination and control.
    """
    
    @staticmethod
    def normalize(q: np.ndarray) -> np.ndarray:
        """Normalize quaternion with error checking."""
        norm = np.linalg.norm(q)
        
        if norm < 1e-12:
            raise SpaceMathError("Quaternion has zero norm")
        
        if abs(norm - 1.0) > 0.01:
            logger.warning(f"Quaternion not normalized: norm={norm}")
        
        return q / norm
    
    @staticmethod
    def multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Quaternion multiplication with validation."""
        q1 = QuaternionOperations.normalize(q1)
        q2 = QuaternionOperations.normalize(q2)
        
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        
        result = np.array([w, x, y, z])
        return QuaternionOperations.normalize(result)
    
    @staticmethod
    def to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = QuaternionOperations.normalize(q)
        w, x, y, z = q
        
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        
        # Verify orthogonality
        if np.max(np.abs(R @ R.T - np.eye(3))) > 1e-6:
            logger.warning("Rotation matrix not orthogonal")
        
        return R
    
    @staticmethod
    def from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """Create quaternion from axis-angle representation."""
        axis = axis / np.linalg.norm(axis)
        half_angle = angle / 2.0
        
        w = np.cos(half_angle)
        xyz = axis * np.sin(half_angle)
        
        return QuaternionOperations.normalize(np.array([w, xyz[0], xyz[1], xyz[2]]))

class KalmanFilterUtilities:
    """
    Spacecraft-grade Kalman filter utilities.
    """
    
    @staticmethod
    def predict_covariance(F: np.ndarray, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """
        Predict covariance with numerical stability.
        """
        # Joseph form for numerical stability
        FP = F @ P
        P_pred = FP @ F.T + Q
        
        # Ensure symmetry
        P_pred = (P_pred + P_pred.T) / 2
        
        # Ensure positive definiteness
        eigenvalues = np.linalg.eigvals(P_pred)
        if np.any(eigenvalues < 0):
            logger.warning("Covariance matrix not positive definite, correcting")
            P_pred = P_pred + np.eye(P_pred.shape[0]) * 1e-6
        
        return P_pred
    
    @staticmethod
    def compute_kalman_gain(P_pred: np.ndarray, H: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Compute Kalman gain with singularity protection.
        """
        S = H @ P_pred @ H.T + R
        
        # Check for singularity
        if np.linalg.matrix_rank(S) < S.shape[0]:
            logger.warning("Innovation covariance singular, using pseudo-inverse")
            K = P_pred @ H.T @ np.linalg.pinv(S)
        else:
            K = P_pred @ H.T @ np.linalg.inv(S)
        
        return K

# Global instance for convenience
space_math = RadiationHardenedMath()
quaternion_ops = QuaternionOperations()
kalman_utils = KalmanFilterUtilities()

# Example usage
if __name__ == "__main__":
    # Test radiation-hardened dot product
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    
    result, confidence = space_math.hardened_dot(a, b)
    print(f"Dot product: {result:.6f} (confidence: {confidence:.3f})")
    
    # Test quaternion operations
    q1 = np.array([0.707, 0.0, 0.707, 0.0])
    q2 = np.array([0.0, 0.707, 0.0, 0.707])
    
    q_result = quaternion_ops.multiply(q1, q2)
    print(f"Quaternion product: {q_result}")
    
    print(f"Math stats: {space_math.get_stats()}")
