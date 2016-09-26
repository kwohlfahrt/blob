# Blob Detection

Blob detection based on laplacian-of-gaussian, to detect localized bright foci
in an image. This is similar to the method used in [scikit-image][skimage-log]
but extended to nD arrays and .tif images.

## Usage

`blob.py` is installed as the primary entry point to output blob locations in
human- and machine-readable formats. It takes a grayscale TIFF image and prints
out blob coordinates in CSV format, for example:

    > blob my_image.tif
    ...
    661 309
    768 309
    382 311
    ...

    
The `--help` option provides details of available options.

`demo.py` is provided in the source repository to give a visual example using
the Hubble Deep Field image (from [scikit-image][skimage]) as sample data.

# Installation

No installation is required, `blob.py` functions as a self-contained executable.

If desired, it can be installed as the executable `blob`, using `setup.py`,
detailed description of installation options can be found in the
[official documentation][setuptools].

## Dependencies

[Python 3][python], [Scipy][scipy], [Numpy][numpy] and [tifffile][tifffile]. All
are available from [PyPI][pypi] and can be installed as described in the
[pip documentation][pip-install]. If necessary, a more up-to-date installer for
`tifffile` is maintained [here](https://github.com/kwohlfahrt/tifffile).

### Demo

The demo script additionally requires [matplotlib][matplotlib], which is also
available through [PyPI][pypi].

[setuptools]: https://docs.python.org/3.3/install/#the-new-standard-distutils
[Python]: https://python.org
[Scipy]: https://scipy.org
[Numpy]: https://www.numpy.org
[tifffile]: http://www.lfd.uci.edu/~gohlke/code/tifffile.py.html
[matplotlib]: http://matplotlib.org
[skimage]: http://scikit-image.org
[skimage-log]: http://scikit-image.org/docs/dev/auto_examples/plot_blob.html#laplacian-of-gaussian-log
[pypi]: https://pypi.python.org/pypi
[pip-install]: https://pip.pypa.io/en/stable/user_guide/#installing-packages
