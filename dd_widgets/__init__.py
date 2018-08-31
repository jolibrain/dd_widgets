import logging

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
