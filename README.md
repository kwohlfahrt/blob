# Blob Detection

Blob detection based on laplacian-of-gaussian, adapted for nD .tif images.

# Installation

No installation is required, `blob.py` functions as a self-contained
executable. A setuptools compatible install script is provided to install the
script as the binary `blob`.

## Dependencies

Python 3, Scipy and Numpy

## Usage

`blob` is installed as the primary entry point to output blob locations in
human- and machine-readable formats. The `--help` option provides usage details.

`demo.py` is provided in the source repository to give a visual example using
the Hubble Deep Field image (from [scikit-image](http://scikit-image.org)) as
sample data. This script also requires [matplotlib](http://matplotlib.org).
