import logging
from datetime import datetime
from pathlib import Path

import ipykernel

import requests
from notebook import notebookapp

from .classification import Classification  # noqa: F401
from .ddcsv import CSV  # noqa: F401
from .detection import Detection  # noqa: F401
from .ocr import OCR  # noqa: F401
from .regression import Regression  # noqa: F401
from .segmentation import Segmentation  # noqa: F401
from .text import Text  # noqa: F401
from .tsne_csv import TSNE_CSV  # noqa: F401
from .tsne_txt import TSNE_Text  # noqa: F401
from .widgets import info_loghandler


def notebook_path() -> Path:
    """Returns the absolute path of current notebook, or just a directory if not
    available. The method only works when the security is token-based or if
    there is no password
    """
    try:
        connection_file = Path(ipykernel.get_connection_file()).stem
    except RuntimeError:
        return Path("./tests.log")

    kernel_id = connection_file.split("-", 1)[1].split(".")[0]

    for srv in notebookapp.list_running_servers():
        try:
            # No token and no password, ahem...
            if srv["token"] == "" and not srv["password"]:
                path = srv["url"] + "api/sessions"
            else:
                path = srv["url"] + "api/sessions?token=" + srv["token"]
            c = requests.get(path)
            for sess in c.json():
                if sess["kernel"]["id"] == kernel_id:
                    return Path(srv["notebook_dir"]) / sess["notebook"]["path"]
        except Exception:
            pass  # There may be stale entries in the runtime directory
    return Path(srv["notebook_dir"])


def logfile(p: Path) -> Path:
    if p.is_dir():
        dirname = p / "logs"
        logname = dirname / f"{datetime.now():%Y-%m-%d}_widgets.log"
    else:
        dirname = p.parent / "logs"
        logname = dirname / (f"{datetime.now():%Y-%m-%d}_" + p.stem + ".log")

    if not dirname.exists():
        dirname.mkdir(parents=True)

    return logname


logfile_name = logfile(notebook_path()).as_posix()

# -- Logging --

fmt = "%(asctime)s:%(msecs)d - %(levelname)s"
fmt += " - {%(filename)s:%(lineno)d} %(message)s"

file_handler = logging.FileHandler(logfile_name)

logging.basicConfig(
    format=fmt,
    level=logging.DEBUG,
    datefmt="%m-%d %H:%M:%S",
    handlers=[file_handler, info_loghandler],
)

info_loghandler.setLevel(logging.INFO)

logging.info(f"Creating {logfile_name} file")
