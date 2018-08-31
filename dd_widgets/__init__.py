import logging

from .classification import Classification  # noqa: F401
from .csv import CSV  # noqa: F401
from .detection import Detection  # noqa: F401
from .ocr import OCR  # noqa: F401
from .segmentation import Segmentation  # noqa: F401
from .text import Text  # noqa: F401
from .widgets import widget_output_handler

# -- Logging --

fmt = "%(asctime)s:%(msecs)d - %(levelname)s"
fmt += " - {%(filename)s:%(lineno)d} %(message)s"

file_handler = logging.FileHandler("widgets.log")

logging.basicConfig(
    format=fmt,
    level=logging.DEBUG,
    datefmt="%m-%d %H:%M:%S",
    handlers=[file_handler, widget_output_handler],
)

logging.info("Creating widgets.log file")
