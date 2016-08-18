#!/usr/bin/env python3

from setuptools import setup

setup(name="blob",
      version="0.0.1",
      description="A tool for blob detection in .tif images",
      py_modules=['blob'],
      entry_points={'console_scripts': ['blob=blob:main']}
)
