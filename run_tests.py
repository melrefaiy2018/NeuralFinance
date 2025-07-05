#!/usr/bin/env python3
"""
Test runner script for Stock Prediction LSTM package.

This script provides convenient commands to run different types of tests.
"""

import subprocess
import sys
import os
import argparse


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {description}:")
        print(f"Return code: {e.returncode}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False


def install_test_dependencies():
    """Install testing dependencies."""
    dependencies = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0", 
        "pytest-mock>=3.10.0",
        "pytest-xdist>=3.0.0",  # For parallel testing
        "pytest-timeout>=2.1.0"  # For timeout handling
    ]
    
    cmd = f"{sys.executable} -m pip install {' '.join(dependencies)}"
    return run_command(cmd, "Installing test dependencies")


def run_unit_tests():
    """Run unit tests."""
    cmd = "python -m pytest tests/unit -v --tb=short"
    return run_command(cmd, "Unit Tests")


def run_integration_tests():
    """Run integration tests."""
    cmd = "python -m pytest tests/integration -v --tb=short"
    return run_command(cmd, "Integration Tests")


def run_all_tests():
    """Run all tests."""
    cmd = "python -m pytest tests -v --tb=short"
    return run_command(cmd, "All Tests")


def run_tests_with_coverage():
    """Run tests with coverage report."""
    cmd = ("python -m pytest tests --cov=stock_prediction_lstm "
           "--cov-report=html --cov-report=term-missing --cov-report=xml")
    return run_command(cmd, "Tests with Coverage")


def run_quick_tests():
    """Run quick tests (excluding slow ones)."""
    cmd = 'python -m pytest tests -v -m "not slow" --tb=short'
    return run_command(cmd, "Quick Tests (excluding slow)")


def run_specific_test_file(test_file):
    """Run tests from a specific file."""
    cmd = f"python -m pytest {test_file} -v --tb=short"
    return run_command(cmd, f"Specific Test File: {test_file}")


def run_tests_parallel():
    """Run tests in parallel."""
    cmd = "python -m pytest tests -v --tb=short -n auto"
    return run_command(cmd, "Parallel Tests")


def check_test_environment():
    """Check if test environment is properly set up."""
    print("Checking test environment...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check if required packages are installed
    required_packages = ["pytest", "pandas", "numpy", "tensorflow"]
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages before running tests.")
        return False
    
    # Check test directory structure
    test_dirs = ["tests", "tests/unit", "tests/integration", "tests/test_data"]
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            print(f"✓ {test_dir} directory exists")
        else:
            print(f"✗ {test_dir} directory is missing")
    
    return len(missing_packages) == 0


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test runner for Stock Prediction LSTM")
    parser.add_argument("--install-deps", action="store_true", 
                       help="Install test dependencies")
    parser.add_argument("--unit", action="store_true", 
                       help="Run unit tests")
    parser.add_argument("--integration", action="store_true", 
                       help="Run integration tests")
    parser.add_argument("--all", action="store_true", 
                       help="Run all tests")
    parser.add_argument("--coverage", action="store_true", 
                       help="Run tests with coverage")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick tests (exclude slow)")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run tests in parallel")
    parser.add_argument("--file", type=str, 
                       help="Run specific test file")
    parser.add_argument("--check-env", action="store_true", 
                       help="Check test environment")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    success = True
    
    if args.check_env:
        success &= check_test_environment()
    
    if args.install_deps:
        success &= install_test_dependencies()
    
    if args.unit:
        success &= run_unit_tests()
    
    if args.integration:
        success &= run_integration_tests()
    
    if args.all:
        success &= run_all_tests()
    
    if args.coverage:
        success &= run_tests_with_coverage()
    
    if args.quick:
        success &= run_quick_tests()
    
    if args.parallel:
        success &= run_tests_parallel()
    
    if args.file:
        success &= run_specific_test_file(args.file)
    
    print(f"\n{'='*60}")
    if success:
        print("✓ All operations completed successfully!")
    else:
        print("✗ Some operations failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
