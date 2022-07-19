from setuptools import setup, find_packages
setup(
    name = 'maestro',
    version = '0.0.1',
    description = 'Multi-fidelity Adaptive Ensemble Stochastic Trust Region Optimization - a plug n play derivate fee solver',
    url = 'https://computing.fnal.gov/hep-on-hpc/',
    author = 'Mohan Krishnamoorthy',
    author_email = 'mkrishnamoorthy2425@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    install_requires = [
        'numpy>=1.15.0',
        'scipy>=1.5.4',
        'matplotlib>=3.0.0',
        # 'mpi4py>=3.0.0',
        'pandas>=1.1.5',
        'pyDOE>=0.3.8',
        'pyDOE2>=1.3.0',
        'apprenticeDFO>=1.0.6'

    ],
    scripts=["bin/maestro-run","maestro/optimizationtask"],
    extras_require = {
    },
    entry_points = {
    },
    dependency_links = [
    ]
)
