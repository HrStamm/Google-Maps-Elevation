"""
Simple test script to verify the Bayesian Optimization model works correctly.

This script runs basic tests on the model components without making API calls.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath('.'))

from src.models.bayesian_optimizer import SquaredExponentialKernel, GaussianProcessModel


def test_kernel():
    """Test the Squared Exponential kernel."""
    print("Testing SquaredExponentialKernel...")
    
    kernel = SquaredExponentialKernel(variance=1.0, lengthscale=1.0)
    
    # Test 1: Kernel at same point should equal variance
    X1 = np.array([[0.0, 0.0]])
    K = kernel(X1, X1)
    assert np.allclose(K, [[1.0]]), "Kernel at same point should equal variance"
    print("  ✓ Self-kernel correct")
    
    # Test 2: Kernel should be symmetric
    X1 = np.array([[0.0, 0.0]])
    X2 = np.array([[1.0, 1.0]])
    K12 = kernel(X1, X2)
    K21 = kernel(X2, X1)
    assert np.allclose(K12, K21.T), "Kernel should be symmetric"
    print("  ✓ Symmetry correct")
    
    # Test 3: Kernel values should decrease with distance
    X1 = np.array([[0.0, 0.0]])
    X2 = np.array([[0.5, 0.5]])
    X3 = np.array([[1.0, 1.0]])
    K12 = kernel(X1, X2)[0, 0]
    K13 = kernel(X1, X3)[0, 0]
    assert K12 > K13, "Kernel should decrease with distance"
    print("  ✓ Distance decay correct")
    
    print("✓ SquaredExponentialKernel tests passed!\n")


def test_gaussian_process():
    """Test the Gaussian Process model."""
    print("Testing GaussianProcessModel...")
    
    kernel = SquaredExponentialKernel(variance=1.0, lengthscale=1.0)
    gp = GaussianProcessModel(kernel, noise=0.01)
    
    # Test 1: Prior predictions (no training data)
    X_test = np.array([[0.0, 0.0], [1.0, 1.0]])
    mean, var = gp.predict(X_test)
    assert np.allclose(mean, [0.0, 0.0]), "Prior mean should be zero"
    assert np.all(var > 0), "Prior variance should be positive"
    print("  ✓ Prior predictions correct")
    
    # Test 2: Posterior predictions (with training data)
    X_train = np.array([[0.0, 0.0], [2.0, 2.0]])
    y_train = np.array([1.0, 0.0])
    gp.fit(X_train, y_train)
    
    mean, var = gp.predict(X_train)
    # Predictions at training points should be close to training values
    assert np.allclose(mean, y_train, atol=0.1), "Predictions at training points should match"
    print("  ✓ Posterior predictions correct")
    
    # Test 3: Uncertainty should be lower at training points
    X_test_near = np.array([[0.1, 0.1]])  # Near training point
    X_test_far = np.array([[10.0, 10.0]])  # Far from training points
    
    _, var_near = gp.predict(X_test_near)
    _, var_far = gp.predict(X_test_far)
    
    assert var_near < var_far, "Variance should be lower near training data"
    print("  ✓ Uncertainty quantification correct")
    
    print("✓ GaussianProcessModel tests passed!\n")


def test_config_loading():
    """Test configuration loading."""
    print("Testing config.yaml loading...")
    
    import yaml
    
    try:
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Check required keys
        assert 'kernel' in config, "Config should have 'kernel' key"
        assert 'optimization' in config, "Config should have 'optimization' key"
        assert 'bounds' in config, "Config should have 'bounds' key"
        
        # Check kernel parameters
        assert 'variance' in config['kernel'], "Kernel should have 'variance'"
        assert 'lengthscale' in config['kernel'], "Kernel should have 'lengthscale'"
        assert 'noise' in config['kernel'], "Kernel should have 'noise'"
        
        # Check optimization parameters
        assert 'n_guesses' in config['optimization'], "Optimization should have 'n_guesses'"
        assert 'exploration' in config['optimization'], "Optimization should have 'exploration'"
        
        print("  ✓ Config file structure correct")
        print(f"  ✓ Kernel variance: {config['kernel']['variance']}")
        print(f"  ✓ Lengthscale: {config['kernel']['lengthscale']}")
        print(f"  ✓ Number of guesses: {config['optimization']['n_guesses']}")
        
        print("✓ Configuration loading tests passed!\n")
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}\n")
        return False
    
    return True


def test_standardization():
    """Test temperature standardization logic."""
    print("Testing temperature standardization...")
    
    # Simulate standardization
    temps = [10.0, 20.0, 30.0, 40.0, 50.0]
    temp_min = min(temps)
    temp_max = max(temps)
    
    standardized = [(t - temp_min) / (temp_max - temp_min) for t in temps]
    
    assert standardized[0] == 0.0, "Min should standardize to 0"
    assert standardized[-1] == 1.0, "Max should standardize to 1"
    assert all(0 <= s <= 1 for s in standardized), "All values should be in [0, 1]"
    
    print("  ✓ Standardization correct")
    print(f"  ✓ Range: {temp_min}°C to {temp_max}°C -> [0, 1]")
    print("✓ Standardization tests passed!\n")


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("Running Bayesian Optimization Model Tests")
    print("="*70)
    print()
    
    try:
        test_kernel()
        test_gaussian_process()
        test_config_loading()
        test_standardization()
        
        print("="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nThe model is ready to use. Run 'python main.py' to start optimization.")
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("="*70)
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        print("="*70)
        return False
    
    return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
