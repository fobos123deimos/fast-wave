import setuptools
from setuptools import find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

name = "fast_wave"
version = "1.1.1"
description = "Package for the calculation of the time-independent wavefunction." 
author_email = "matheusgomescord@gmail.com"
url = "https://github.com/pikachu123deimos/fast-wave" 

install_requires = [
    "llvmlite==0.42.0",
    "mpmath==1.3.0",
    "numba==0.59.1",
    "numpy==1.26.4",
    "scipy==1.13.0",
    "sympy==1.12",
]

test_requires = [
    "pytest= ^7.1.2",
]

packages = find_packages(where='src')

classifiers = [
    'Programming Language :: Python :: 3.11',  
    'Topic :: Scientific/Engineering :: Mathematics',  
    'License :: OSI Approved :: BSD License',  
    'Development Status :: 5 - Production/Stable', 
]

setuptools.setup(
    name=name,
    version=version,
    long_description=long_description,
    long_description_content_type="text/markdown",
    description=description,
    author=name,
    author_email=author_email,
    url=url,
    install_requires=install_requires,
    test_requires=test_requires,
    packages=packages,
    package_dir={'': 'src'},
    classifiers=classifiers,
)
