#!/usr/bin/env python3
"""Setup script for stock_prediction_lstm package."""

from setuptools import setup, find_packages
import os


# Read version from __version__.py
def read_version():
    version_file = os.path.join(
        os.path.dirname(__file__), "stock_prediction_lstm", "__version__.py"
    )
    with open(version_file, "r") as f:
        exec(f.read())
    return locals()["version"]


# Read requirements from requirements.txt
def read_requirements():
    requirements_file = os.path.join(os.path.dirname(__file__), "requirements.txt")
    with open(requirements_file, "r") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


# Read long description from README if it exists
def read_long_description():
    readme_file = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_file):
        with open(readme_file, "r", encoding="utf-8") as f:
            return f.read()
    return ""


# Read requirements for development
def read_dev_requirements():
    dev_requirements = [
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
        "black>=22.0.0",
        "flake8>=5.0.0",
        "mypy>=1.0.0",
        "pre-commit>=2.20.0",
        "sphinx>=5.0.0",
        "sphinx-rtd-theme>=1.0.0",
        "jupyter>=1.0.0",
        "notebook>=6.5.0",
    ]
    return dev_requirements


setup(
    name="stock-prediction-lstm",
    version=read_version(),
    author="Mohamed",
    author_email="mohamed@example.com",  # Replace with your actual email
    description="A comprehensive stock prediction system using LSTM neural networks with sentiment analysis and technical indicators",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock_prediction_lstm",  # Replace with your actual repository
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/stock_prediction_lstm/issues",
        "Documentation": "https://yourusername.github.io/stock_prediction_lstm/",
        "Source Code": "https://github.com/yourusername/stock_prediction_lstm",
        "Changelog": "https://github.com/yourusername/stock_prediction_lstm/blob/main/CHANGELOG.md",
    },
    packages=find_packages(exclude=["tests*", "examples*", "docs*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: Jupyter",
        "Environment :: Console",
        "Environment :: Web Environment",
    ],
    keywords="stock prediction, lstm, neural networks, sentiment analysis, technical analysis, machine learning, finance, trading, deep learning",
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": read_dev_requirements(),
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "sphinx-autodoc-typehints>=1.19.0",
            "myst-parser>=0.18.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipywidgets>=8.0.0",
            "jupyterlab>=3.4.0",
        ],
        "web": [
            "streamlit>=1.25.0",
            "dash>=2.10.0",
            "plotly>=5.15.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.8.0",
        ],
    },
    include_package_data=True,
    package_data={
        "stock_prediction_lstm": [
            "config/**/*",
            "data_cache/**/*",
            "web/static/**/*",
            "web/templates/**/*",
        ],
    },
    entry_points={
        "console_scripts": [
            "stock-predict=stock_prediction_lstm.cli.main:main",
        ],
    },
    zip_safe=False,
    platforms=["any"],
    license="MIT",
    test_suite="tests",
    tests_require=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pytest-mock>=3.10.0",
    ],
    # Additional metadata for PyPI
    download_url="https://github.com/yourusername/stock_prediction_lstm/archive/v1.0.0.tar.gz",
    # Ensure wheel is built properly
    options={
        "bdist_wheel": {
            "universal": False,
        },
    },
)
