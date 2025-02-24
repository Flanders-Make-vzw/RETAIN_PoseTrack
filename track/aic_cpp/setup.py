# filepath: /home/track/aic_cpp/setup.py
from setuptools import setup, find_packages

setup(
    name='aic_cpp',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'pybind11',
        # other dependencies
    ],
    # other setup parameters
)