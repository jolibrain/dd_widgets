import random
from heapq import nlargest
from pathlib import Path
from tempfile import mkstemp
from typing import Iterator, Optional, Tuple, TypeVar
import re

import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib import patches
from matplotlib.cm import get_cmap

import cv2

Elt = TypeVar("Elt")


def sample_from_iterable(it: Iterator[Elt], k: int) -> Iterator[Elt]:
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


def is_url(path):
    regex = re.compile(
        r'^(?:http|ftp)s?://' # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
        r'localhost|' #localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
        r'(?::\d+)?' # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE
    )
    return re.match(regex, path) is not None


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
    if data.shape[2] == 3:
        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    elif data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2RGB)
    _, fname = mkstemp(suffix=".png")
    fig, ax = plt.subplots()
    ax.imshow(data)
    if segmentation is not None:
        data = cv2.imread(segmentation.as_posix(), cv2.IMREAD_UNCHANGED)
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
