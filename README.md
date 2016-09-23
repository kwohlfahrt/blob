# Blob Detection

Blob detection based on laplacian-of-gaussian, adapted for nD .tif images.

# Installation

No installation is required, `blob.py` functions as a self-contained executable.
A setuptools compatible install script is provided to install the script as the
binary `blob`, detailed instructions can be found in the
[official documentation][setuptools].

## Dependencies

[Python 3][python], [Scipy][scipy], [Numpy][numpy] and [tifffile][tifffile]. All
are available on PyPI, though a more up-to-date installer for `tifffile` is
maintained [here](https://github.com/kwohlfahrt/tifffile).

### Demo

The demo script additionally requires [matplotlib][matplotlib].

## Usage

`blob` is installed as the primary entry point to output blob locations in
human- and machine-readable formats. The `--help` option provides usage details.

`demo.py` is provided in the source repository to give a visual example using
the Hubble Deep Field image (from [scikit-image][skimage]) as sample data.

[setuptools]: https://docs.python.org/3.3/install/#the-new-standard-distutils
[Python]: https://python.org
[Scipy]: https://scipy.org
[Numpy]: https://www.numpy.org
[tifffile]: http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
[matplotlib]: http://matplotlib.org
[skimage]: http://scikit-image.org
