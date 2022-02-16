from setuptools import setup, find_packages
setup(
    name = 'mfstrodf',
    version = '0.0.1',
    description = 'Multi fidelity Derivative-free Stochastic Trust Region Optimization',
    url = 'https://computing.fnal.gov/hep-on-hpc/',
    author = 'Mohan Krishnamoorthy',
    author_email = 'mkrishnamoorthy2425@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'numpy>=1.15.0',
        'scipy>=1.7.2',
        'matplotlib>=3.0.0',
        'mpi4py>=3.0.0',
        'pandas>=1.4.0',

    ],
    scripts=["bin/mfstrodf-run"],
    extras_require = {
    },
    entry_points = {
    },
    dependency_links = [
    ]
)
