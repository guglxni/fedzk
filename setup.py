#!/usr/bin/env python3
"""
FEDZK Setup Script

Setup script for FEDZK package distribution.
This script provides backward compatibility and additional configuration options.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def get_version():
    """Get version from VERSION file or pyproject.toml."""
    version_file = Path(__file__).parent / 'VERSION'
    if version_file.exists():
        with open(version_file, 'r') as f:
            return f.read().strip()

    # Fallback to pyproject.toml
    try:
        import tomli
        pyproject_file = Path(__file__).parent / 'pyproject.toml'
        if pyproject_file.exists():
            with open(pyproject_file, 'rb') as f:
                data = tomli.load(f)
                return data.get('project', {}).get('version', '0.1.0')
    except ImportError:
        pass

    return '0.1.0'

def get_long_description():
    """Get long description from README.md."""
    readme_file = Path(__file__).parent / 'README.md'
    if readme_file.exists():
        with open(readme_file, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

def get_requirements():
    """Get requirements from pyproject.toml."""
    requirements = []

    try:
        import tomli
        pyproject_file = Path(__file__).parent / 'pyproject.toml'
        if pyproject_file.exists():
            with open(pyproject_file, 'rb') as f:
                data = tomli.load(f)
                project_data = data.get('project', {})
                dependencies = project_data.get('dependencies', [])
                requirements.extend(dependencies)
    except ImportError:
        # Fallback to basic requirements
        requirements = [
            'numpy>=1.21.0',
            'pandas>=1.3.0',
            'cryptography>=3.4.0',
            'fastapi>=0.68.0',
            'uvicorn>=0.15.0',
            'pydantic>=1.8.0',
            'sqlalchemy>=1.4.0',
            'redis>=4.0.0'
        ]

    return requirements

# Get package metadata
version = get_version()
long_description = get_long_description()
requirements = get_requirements()

# Setup configuration
setup(
    name='fedzk',
    version=version,
    description='Federated Learning with Zero-Knowledge Proofs',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='FEDZK Team',
    author_email='support@fedzk.io',
    url='https://github.com/fedzk/fedzk',
    project_urls={
        'Documentation': 'https://docs.fedzk.io',
        'Source': 'https://github.com/fedzk/fedzk',
        'Tracker': 'https://github.com/fedzk/fedzk/issues',
        'Community': 'https://community.fedzk.io',
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Financial and Insurance Industry',
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Security :: Cryptography',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=[
        'federated-learning',
        'zero-knowledge-proofs',
        'privacy-preserving',
        'machine-learning',
        'cryptography',
        'zkp',
        'privacy',
        'secure-computation'
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-asyncio>=0.18.0',
            'pytest-mock>=3.7.0',
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            'pre-commit>=2.17.0',
            'coverage[toml]>=6.0.0',
        ],
        'gpu': [
            'torch>=1.12.0',
            'torchvision>=0.13.0',
            'torchaudio>=0.12.0',
            'tensorflow>=2.9.0',
            'xgboost>=1.6.0',
            'lightgbm>=3.3.0',
            'catboost>=1.0.0',
        ],
        'docs': [
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            'sphinx-autodoc-typehints>=1.18.0',
            'myst-parser>=0.17.0',
            'sphinx-copybutton>=0.5.0',
            'sphinx-design>=0.2.0',
        ],
        'all': [
            'fedzk[dev,gpu,docs]',
        ]
    },
    entry_points={
        'console_scripts': [
            'fedzk=fedzk.cli:app',
            'fedzk-admin=fedzk.cli:admin_app',
            'fedzk-coordinator=fedzk.cli:coordinator_app',
            'fedzk-worker=fedzk.cli:worker_app',
        ],
    },
    include_package_data=True,
    package_data={
        'fedzk': [
            'circuits/*',
            'config/*.yaml',
            'templates/*',
            'static/*',
            'migrations/*',
        ],
    },
    data_files=[
        ('config', ['config/fedzk.yaml.example']),
        ('docs', ['README.md', 'CHANGELOG.md']),
    ],
    zip_safe=False,
    platforms=['any'],
    license='MIT',
    license_files=['LICENSE'],
)
