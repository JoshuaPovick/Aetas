#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name="aetas",
      version='0.0.1',
      description='Stellar Ages',
      author='Joshua Povick, David Nidever',
      author_email='joshua.povick@montana.edu',
      url='https://github.com/JoshuaPovick/aetas',
      packages=find_packages(exclude=["tests"]),
      scripts=['bin/aetas'],
      requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils']
#      include_package_data=True,
)
