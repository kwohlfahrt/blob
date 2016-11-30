#!/usr/bin/env python3

from numpy import zeros, ones, asarray
from numpy.linalg import norm
from math import pi
from scipy.ndimage.filters import gaussian_laplace

def localMinima(data, threshold):
    from numpy import ones, roll, nonzero, transpose

    if threshold is not None:
        peaks = data < threshold
    else:
        peaks = ones(data.shape, dtype=data.dtype)

    for axis in range(data.ndim):
        peaks &= data <= roll(data, -1, axis)
        peaks &= data <= roll(data, 1, axis)
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
    if positions.shape[1] == 3:
        intersections = sphereIntersection(radii, radii.T, distances)
        volumes = 4/3 * pi * radii ** 3
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
    if args.outfile is not None:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    image = imread(str(args.image))
    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')

    if args.axes is not None:
        image = image.sum(tuple(args.axes))

    if args.peaks.suffix == ".csv":
        peaks = loadtxt(str(args.peaks))[:, ::-1]
    elif args.peaks.suffix == ".pickle":
        with args.peaks.open("rb") as f:
            peaks = load(f)

    peaks = peaks / scale
    peaks = delete(peaks, args.axes, axis=1)

    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.scatter(*peaks.T[::-1], edgecolor='red', facecolor='none')
    if args.outfile is None:
        plt.show()
    else:
        fig.savefig(str(args.outfile))

def find(args):
    from sys import stdout
    from tifffile import imread

    image = imread(str(args.image)).astype('float32')
    if args.axes is not None:
        image = image.sum(tuple(args.axes))

    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')
    blobs = findBlobs(image, range(*args.size), args.threshold)
    blobs = blobs[peakEnclosed(blobs[:, 1:], shape=image.shape, size=args.edge)]
    if args.format == "csv":
        import csv

        writer = csv.writer(stdout, delimiter=' ')
        for blob in blobs:
            writer.writerow(blob[1:][::-1] * scale)
    elif args.format == "pickle":
        from pickle import dump, HIGHEST_PROTOCOL
        from functools import partial
        dump = partial(dump, protocol=HIGHEST_PROTOCOL)

        dump(blobs[:, 1:] * scale, stdout.buffer)

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
    find_parser.add_argument("--format", choices={"csv", "pickle"}, default="csv",
                             help="The output format (for stdout)")
    find_parser.add_argument("--edge", type=int, default=0,
                             help="Minimum distance to edge allowed.")
    find_parser.set_defaults(func=find)

    plot_parser = subparsers.add_parser("plot")
    plot_parser.add_argument("image", type=Path, help="The image to process")
    plot_parser.add_argument("peaks", type=Path, help="The peaks to plot")
    plot_parser.add_argument("--outfile", type=Path,
                             help="Where to save the plot (omit to display)")
    plot_parser.set_defaults(func=plot)

    for p in (plot_parser, find_parser):
        p.add_argument("--axes", type=int, nargs='+', default=None,
                       help="The projection axes")
        p.add_argument("--scale", nargs="*", type=float,
                       help="The scale for the points along each axis.")

    args = parser.parse_args(argv[1:] if args is None else args)
    args.func(args)

if __name__ == "__main__":
    main()
