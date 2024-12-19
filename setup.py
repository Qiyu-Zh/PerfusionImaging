from setuptools import setup, find_packages
import os

setup(
    name='PerfusionImaging',
    version="0.1.0",
    description="A short description of your package",
    packages=find_packages(),  # Automatically find and include your package
    install_requires=[         # List of dependencies
      "SimpleITK",
      "ipywidgets",
      "IPython",
      "scipy",
      "matplotlib",
      "numpy",
      "pydicom",
      "antspyx"
    ],
    classifiers=[              # Metadata
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
)
