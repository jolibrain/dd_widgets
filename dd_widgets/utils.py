import random
from heapq import nlargest
from pathlib import Path
from tempfile import mkstemp
from typing import Iterator, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib import patches
from matplotlib.cm import get_cmap

import cv2

Elt = TypeVar("Elt")


def sample_from_iterable(it: Iterator[Elt], k: int) -> Iterator[Elt]:
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


def img_handle(
    path: Path,
    segmentation: Optional[Path] = None,
    bbox: Optional[Path] = None,
    nclasses: int = -1,
    imread_args: tuple = tuple(),
) -> Tuple[Tuple[int, ...], Image]:

    if not path.exists():
        raise ValueError("File {} does not exist".format(path))
    data = cv2.imread(path.as_posix(), *imread_args)
    _, fname = mkstemp(suffix=".png")
    fig, ax = plt.subplots()
    ax.imshow(data)
    if segmentation is not None:
        data = cv2.imread(segmentation.as_posix(), *imread_args)
        ax.imshow(data, alpha=.8)
        if data.max() >= nclasses > -1:
            raise RuntimeError(
                "Index {max} present in {filename}".format(
                    max=data.max(), filename=segmentation.as_posix()
                )
            )
    if bbox is not None:

        if nclasses > -1:
            cmap = get_cmap("jet", nclasses - 1)

        with bbox.open("r") as fh:
            for line in fh.readlines():
                tag, xmin, ymin, xmax, ymax = (
                    int(float(x)) for x in line.strip().split()
                )
                if tag >= nclasses > -1:
                    raise RuntimeError(
                        "Index {max} present in {filename}".format(
                            max=tag, filename=bbox.as_posix()
                        )
                    )
                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor=cmap(tag) if nclasses > -1 else "blue",
                    facecolor="none",
                )
                ax.add_patch(rect)

    fig.savefig(fname)
    plt.close(fig)
    return data.shape, Image(fname)
