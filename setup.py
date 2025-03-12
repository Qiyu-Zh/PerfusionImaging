from setuptools import setup, find_packages

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
      "numpy==1.25",
      "pydicom",
      "antspyx==0.4.2",
      "scikit-image",
      "shapely",
      "tqdm"
    ],
    classifiers=[              # Metadata
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
)
