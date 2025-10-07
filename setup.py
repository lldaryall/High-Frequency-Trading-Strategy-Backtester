"""
Setup script for Flashback HFT backtesting engine with Cython extensions.
"""

from setuptools import setup, Extension, find_packages
import os

# Check if Cython is available
try:
    from Cython.Build import cythonize
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False
    print("Warning: Cython not available. C extensions will not be built.")
    print("Install Cython with: pip install cython")

# Define Cython extensions
extensions = []

if CYTHON_AVAILABLE:
    try:
        import numpy as np
        extensions = [
            Extension(
                "flashback.market._match",
                sources=["flashback/market/_match.pyx"],
                include_dirs=[np.get_include()],
                extra_compile_args=["-O3", "-ffast-math"],
                extra_link_args=["-O3"],
            )
        ]
    except ImportError:
        print("Warning: NumPy not available. C extensions will not be built.")
        print("Install NumPy with: pip install numpy")
        extensions = []

# Setup configuration
setup(
    name="flashback",
    version="0.1.0",
    description="High-Frequency Trading Strategy Backtester",
    author="Flashback Team",
    packages=find_packages(),
    ext_modules=cythonize(extensions, compiler_directives={
        'language_level': 3,
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }) if CYTHON_AVAILABLE else [],
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyarrow>=12.0.0",
        "pyyaml>=6.0",
        "click>=8.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "pydantic>=2.0.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "dev": [
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
        ],
        "cython": [
            "cython>=0.29.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "flashback=flashback.cli.main:cli",
        ],
    },
    python_requires=">=3.11",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Cython",
    ],
    zip_safe=False,  # Required for Cython extensions
)
