#!/usr/bin/env python3

from numpy import zeros, ones, asarray
from numpy.linalg import norm
from math import pi

try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
except ImportError:
    from numpy.fft import fftn, ifftn

def gaussianPsf(shape, sigmas, dtype='float'):
    from math import sqrt
    from numpy import ones, ogrid, exp

    output = ones(shape, dtype=dtype)
    grids = ogrid[tuple(slice(None, s) for s in shape)]
    for grid, sigma in zip(grids, sigmas):
        # Pyramid function with integer operations
        grid = (grid.size - abs(grid * 2 - grid.size)) // 2
        values = 1 / (sigma * sqrt(2 * pi)) * exp(-( grid / sigma ) ** 2 / 2)
        output *= values
    return output

def gaussianPsfHat(shape, sigmas, dtype='float'):
    from numpy import ones, ogrid, exp

    output = ones(shape, dtype=dtype)
    grids = ogrid[tuple(slice(None, s) for s in shape)]
    for grid, sigma in zip(grids, sigmas):
        # Pyramid function with integer operations
        grid = (grid.size - abs(grid * 2 - grid.size)) // 2
        values = exp(-( grid * pi * sigma / grid.size ) ** 2 * 2)
        output *= values
    return output

# Circular gradient. Add more padding methods?
def gradient(data, axis, backward=False):
    from numpy import roll
    if backward:
        return data - roll(data, -1, axis)
    else:
        return roll(data, 1, axis) - data

def div(*data):
    return sum(gradient(d, axis) for axis, d in enumerate(data))

def laplacianOperator(data):
    return div(*(gradient(data, a, backward=True) for a in range(data.ndim)))

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
    data_hat = fftn(data)

    log = empty((len(scales),) + data.shape, dtype=data.dtype)
    for slog, scale in zip(log, scales):
        blur = gaussianPsfHat(data.shape, repeat(scale), data.dtype)
        blurred = ifftn(data_hat * blur).real
        slog[...] = scale ** 2 * laplacianOperator(blurred)

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
    from numpy import ones, triu

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
    return peaks[~delete]

if __name__ == '__main__':
    from argparse import ArgumentParser
    from pathlib import Path
    import sys

    from tifffile import imread

    parser = ArgumentParser(description="Print a list of spots from an image")
    parser.add_argument("image", type=Path, help="The image to process")
    parser.add_argument("--size", type=int, nargs=2, default=(1, 1),
                        help="The range of sizes (in px) to search.")
    parser.add_argument("--threshold", type=float, default=5,
                        help="The minimum spot intensity")
    parser.add_argument("--plot", action="store_true",
                        help="Plot the extracted points on a maximum intensity projection.")
    parser.add_argument("--scale", nargs="*", type=float,
                       help="The scale for the points along each axis.")
    parser.add_argument("--format", choices={"csv", "pickle"}, default="csv",
                        help="The output format (for stdout)")

    args = parser.parse_args()

    image = imread(str(args.image)).astype('float32')
    scale = asarray(args.scale) if args.scale else ones(image.ndim, dtype='int')
    blobs = findBlobs(image, range(*args.size), args.threshold)
    if args.format == "csv":
        import csv

        writer = csv.writer(sys.stdout, delimiter=' ')
        for blob in blobs:
            writer.writerow(blob[1:][::-1] * scale)
    elif args.format == "pickle":
        from pickle import dump, HIGHEST_PROTOCOL
        from functools import partial
        dump = partial(dump, protocol=HIGHEST_PROTOCOL)

        dump(blobs[:, 1:] * scale, sys.stdout.buffer)

    if args.plot:
        import matplotlib.pyplot as plt
        from numpy import amax

        proj = amax(image, axis=tuple(range(image.ndim - 2)))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(proj, cmap="gray", interpolation="nearest")
        ax.scatter(blobs[:, -1], blobs[:, -2], color="red", marker="+")
        plt.show()
