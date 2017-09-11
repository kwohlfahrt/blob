#!/usr/bin/env python3

from numpy import zeros, ones, asarray
from numpy.linalg import norm
from math import pi
from scipy.ndimage.filters import gaussian_laplace, minimum_filter
from operator import contains
from functools import partial
from itertools import filterfalse

def localMinima(data, threshold):
    from numpy import ones, nonzero, transpose

    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)

    peaks &= data == minimum_filter(data, size=(3,) * data.ndim)
    return transpose(nonzero(peaks))

def blobLOG(data, scales=range(1, 10, 1), threshold=-30):
    """Find blobs. Returns [[scale, x, y, ...], ...]"""
    from numpy import empty, asarray
    from itertools import repeat

    data = asarray(data)
    scales = asarray(scales)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        slog[...] = scale ** 2 * gaussian_laplace(data, scale)

    peaks = localMinima(log, threshold=threshold)
    peaks[:, 0] = scales[peaks[:, 0]]
    return peaks

def sphereIntersection(r1, r2, d):
    # https://en.wikipedia.org/wiki/Spherical_cap#Application

    valid = (d < (r1 + r2)) & (d > 0)
    return (pi * (r1 + r2 - d) ** 2
            * (d ** 2 + 6 * r2 * r1
               + 2 * d * (r1 + r2)
               - 3 * (r1 - r2) ** 2)
            / (12 * d)) * valid

def circleIntersection(r1, r2, d):
    # http://mathworld.wolfram.com/Circle-CircleIntersection.html
    from numpy import arccos, sqrt

    return (r1 ** 2 * arccos((d ** 2 + r1 ** 2 - r2 ** 2) / (2 * d * r1))
            + r2 ** 2 * arccos((d ** 2 + r2 ** 2 - r1 ** 2) / (2 * d * r2))
            - sqrt((-d + r1 + r2) * (d + r1 - r2)
                   * (d - r1 + r2) * (d + r1 + r2)) / 2)

def findBlobs(img, scales=range(1, 10), threshold=30, max_overlap=0.05):
    from numpy import ones, triu, seterr
    old_errs = seterr(invalid='ignore')

    peaks = blobLOG(img, scales=scales, threshold=-threshold)
    radii = peaks[:, 0]
    positions = peaks[:, 1:]

    distances = norm(positions[:, None, :] - positions[None, :, :], axis=2)

    if positions.shape[1] == 2:
        intersections = circleIntersection(radii, radii.T, distances)
        volumes = pi * radii ** 2
    elif positions.shape[1] == 3:
        intersections = sphereIntersection(radii, radii.T, distances)
        volumes = 4/3 * pi * radii ** 3
    else:
        raise ValueError("Invalid dimensions for position ({}), need 2 or 3."
                         .format(positions.shape[1]))

    delete = ((intersections > (volumes * max_overlap))
              # Remove the smaller of the blobs
              & ((radii[:, None] < radii[None, :])
                 # Tie-break
                 | ((radii[:, None] == radii[None, :])
                    & triu(ones((len(peaks), len(peaks)), dtype='bool'))))
    ).any(axis=1)

    seterr(**old_errs)
    return peaks[~delete]

def peakEnclosed(peaks, shape, size=1):
    from numpy import asarray

    shape = asarray(shape)
    return ((size <= peaks).all(axis=-1) & (size < (shape - peaks)).all(axis=-1))

def plot(args):
    from tifffile import imread
    from numpy import loadtxt, delete
    from pickle import load
    import matplotlib
    from mpl_toolkits.axes_grid.anchored_artists import AnchoredAuxTransformBox
    from matplotlib.text import Text
    from matplotlib.text import Line2D

    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image = imread(str(args.image)).T
    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')

    if args.peaks.suffix == '.txt':
        peaks = loadtxt(str(args.peaks), ndmin=2)
    elif args.peaks.suffix == ".csv":
        peaks = loadtxt(str(args.peaks), ndmin=2, delimiter=',')
    elif args.peaks.suffix == ".pickle":
        with args.peaks.open("rb") as f:
            peaks = load(f)
    else:
        raise ValueError("Unrecognized file type: '{}', need '.pickle' or '.csv'"
                         .format(args.peaks.suffix))
    peaks = peaks / scale

    proj_axes = tuple(filterfalse(partial(contains, args.axes), range(image.ndim)))
    image = image.max(proj_axes)
    peaks = delete(peaks, proj_axes, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=args.size)
    ax.imshow(image.T, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(*peaks.T, edgecolor="C1", facecolor='none')

    if args.scalebar is not None:
        pixel, units, length = args.scalebar
        pixel = float(pixel)
        length = int(length)

        box = AnchoredAuxTransformBox(ax.transData, loc=4)
        box.patch.set_alpha(0.8)
        bar = Line2D([-length/pixel/2, length/pixel/2], [0.0, 0.0], color='black')
        box.drawing_area.add_artist(bar)
        label = Text(
            0.0, 0.0, "{} {}".format(length, units),
            horizontalalignment="center", verticalalignment="bottom"
        )
        box.drawing_area.add_artist(label)
        ax.add_artist(box)

    if args.outfile is None:
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig(str(args.outfile))

def find(args):
    from sys import stdout
    from tifffile import imread

    image = imread(str(args.image)).astype('float32')

    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')
    blobs = findBlobs(image, range(*args.size), args.threshold)[:, 1:] # Remove scale
    blobs = blobs[peakEnclosed(blobs, shape=image.shape, size=args.edge)]
    blobs = blobs[:, ::-1] # Reverse to xyz order
    blobs = blobs * scale

    if args.format == "pickle":
        from pickle import dump, HIGHEST_PROTOCOL
        from functools import partial
        dump = partial(dump, protocol=HIGHEST_PROTOCOL)

        dump(blobs, stdout.buffer)
    else:
        import csv

        if args.format == 'txt':
            delimiter = ' '
        elif args.format == 'csv':
            delimiter = ','
        writer = csv.writer(stdout, delimiter=delimiter)
        for blob in blobs:
            writer.writerow(blob)

# For setuptools entry_points
def main(args=None):
    from argparse import ArgumentParser
    from pathlib import Path
    from sys import argv

    parser = ArgumentParser(description="Find peaks in an nD image")
    subparsers = parser.add_subparsers()

    find_parser = subparsers.add_parser("find")
    find_parser.add_argument("image", type=Path, help="The image to process")
    find_parser.add_argument("--size", type=int, nargs=2, default=(1, 1),
                             help="The range of sizes (in px) to search.")
    find_parser.add_argument("--threshold", type=float, default=5,
                             help="The minimum spot intensity")
    find_parser.add_argument("--format", choices={"csv", "txt", "pickle"}, default="csv",
                             help="The output format (for stdout)")
    find_parser.add_argument("--edge", type=int, default=0,
                             help="Minimum distance to edge allowed.")
    find_parser.set_defaults(func=find)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("image", type=Path, help="The image to process")
    plot_parser.add_argument("peaks", type=Path, help="The peaks to plot")
    plot_parser.add_argument("outfile", nargs='?', type=Path, default=None,
                             help="Where to save the plot (omit to display)")
    plot_parser.add_argument("--axes", type=int, nargs=2, default=(0, 1),
                             help="The axes to plot")
    plot_parser.add_argument("--size", type=float, nargs=2, default=(5, 5),
                             help="The size of the figure (in inches)")
    plot_parser.add_argument("--scalebar", type=str, nargs=3, default=None,
                             help="The pixel-size, units and scalebar size")
    plot_parser.set_defaults(func=plot)

    for p in (plot_parser, find_parser):
        p.add_argument("--scale", nargs="*", type=float,
                       help="The scale for the points along each axis.")

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)

if __name__ == "__main__":
    main()
