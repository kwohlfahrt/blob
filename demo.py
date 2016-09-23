#!/usr/bin/env python3

from matplotlib import pyplot as plt
import blob
from pathlib import Path
from tifffile import imread

data_path = Path(__file__).parent / "tests" / "fixtures" / "hubble_deep_field.tif"
data = imread(str(data_path)).astype('float32').sum(axis=-1)

# Equivalent to `blobs --size 5 10 --threshold 10 hubble_deep_field.tif`
blobs = blob.findBlobs(data, scales=range(5, 10), threshold=10)

fig, ax = plt.subplots(1, 1)
ax.imshow(data, cmap='gray')
ax.scatter(*blobs.T[1:][::-1], s=blobs.T[0] * 10, edgecolor='red', facecolor='none')
fig.tight_layout()
plt.show()
