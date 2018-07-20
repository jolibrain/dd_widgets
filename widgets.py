# fmt: off

import json
import logging
import random
from heapq import nlargest
from pathlib import Path
from tempfile import mkstemp
from typing import (Any, Dict, Iterator, List, Optional, Tuple, TypeVar,
                    get_type_hints)

import matplotlib.pyplot as plt
from IPython.display import Image, display
from matplotlib import patches

import cv2
import requests
from core import send_dd
from ipywidgets import (HTML, Button, Checkbox, FloatText, HBox, IntText,
                        Layout, Output, SelectMultiple, Text, VBox)

# fmt: on

# -- Logging --

# This should not stay here in the long run
# It's just a convenient way to debug when messages cannot always be printed

logger = logging.getLogger()
logger.setLevel(logging.INFO)

fmt = "%(asctime)s:%(msecs)d - %(levelname)s"
fmt += " - {%(filename)s:%(lineno)d} %(message)s"

logging.basicConfig(
    level=logging.DEBUG,
    format=fmt,
    datefmt="%m-%d %H:%M:%S",
    filename="widgets.log",
    filemode="a",
)

logging.info("Creating widgets.log file")

# -- Basic tools --


def img_handle(
    path: Path, segmentation: Optional[Path] = None, bbox: Optional[Path] = None
) -> Tuple[Tuple[int, ...], Image]:
    data = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    _, fname = mkstemp(suffix=".png")
    fig, ax = plt.subplots()
    ax.imshow(data)
    if segmentation is not None:
        data = cv2.imread(segmentation.as_posix(), cv2.IMREAD_UNCHANGED)
        ax.imshow(data, alpha=.2)
    if bbox is not None:
        with bbox.open("r") as fh:
            for line in fh.readlines():
                tag, xmin, ymin, xmax, ymax = (
                    int(x) for x in line.strip().split()
                )
                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                )
                ax.add_patch(rect)

    fig.savefig(fname)
    plt.close(fig)
    return data.shape, Image(fname)


Elt = TypeVar("Elt")


def sample_from_iterable(it: Iterator[Elt], k: int) -> Iterator[Elt]:
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


# -- Core 'abstract' widget for many tasks


class MLWidget(object):

    _fields = {  # typing: Dict[str, str]
        "sname": "Model name",
        "training_repo": "Training directory",
        "testing_repo": "Testing directory",
    }

    _widget_type = {int: IntText, float: FloatText, bool: Checkbox}

    output = Output(layout=Layout(max_width="650px"))  # typing: Output
    # host: Text
    # port: Text

    def __init__(
        self, sname: str, params: Dict[str, Tuple[Any, type]], *args
    ) -> None:
        self.sname = sname

        self.run_button = Button(description="Run")
        self.info_button = Button(description="Info")
        self.clear_button = Button(description="Clear")

        self._widgets = [  # typing: List[Widget]
            HTML(
                value="<h2>{task} task: {sname}</h2>".format(
                    task=self.__class__.__name__, sname=self.sname
                )
            ),
            self.run_button,
            self.info_button,
            self.clear_button,
        ]

        self.run_button.on_click(self.run)
        self.info_button.on_click(self.info)
        self.clear_button.on_click(self.clear)

        for name, (value, type_hint) in params.items():
            self._add_widget(name, value, type_hint)

        self._configuration = VBox(
            self._widgets, layout=Layout(min_width="250px")
        )

    def _add_widget(self, name, value, type_hint):

        widget_type = self._widget_type.get(type_hint, None)

        if widget_type is None:
            setattr(
                self,
                name,
                Text(  # Widget type by default then convert to str
                    value="" if value is None else str(value),
                    description=self._fields.get(name, name),
                ),
            )
        else:
            setattr(
                self,
                name,
                widget_type(
                    value=type_hint() if value is None else type_hint(value),
                    description=self._fields.get(name, name),
                ),
            )

        self._widgets.append(getattr(self, name))

    def _ipython_display_(self):
        self._main_elt._ipython_display_()

    @output.capture(clear_output=True)
    def clear(self, *_):
        request = "http://{host}:{port}/services/{sname}?clear=full".format(
            host=self.host.value, port=self.port.value, sname=self.sname
        )
        c = requests.delete(request)
        logging.info(
            "Clearing (full) service {sname}: {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )
        print(json.dumps(c.json(), indent=2))

    @output.capture(clear_output=True)
    def info(self, *_):
        # TODO job number
        request = (
            "http://{host}:{port}/train?service={sname}&"
            "job=1&timeout=10".format(
                host=self.host.value, port=self.port.value, sname=self.sname
            )
        )
        c = requests.get(request)
        logging.info(
            "Getting info for service {sname}: {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )
        print(json.dumps(c.json(), indent=2))

    @output.capture(clear_output=True)
    def update_label_list(self, _):
        if self.training_repo.value != "":
            self.train_labels.options = tuple(
                sorted(f.stem for f in Path(self.training_repo.value).glob("*"))
            )
        if self.testing_repo.value != "":
            self.test_labels.options = tuple(
                sorted(f.stem for f in Path(self.testing_repo.value).glob("*"))
            )

        self.train_labels.rows = min(10, len(self.train_labels.options))
        self.test_labels.rows = min(10, len(self.test_labels.options))
        if self.nclasses.value == -1:
            self.nclasses.value = str(len(self.train_labels.options))


class Classification(MLWidget):
    @MLWidget.output.capture(clear_output=True)
    def update_train_file_list(self, *args):
        if len(self.train_labels.value) == 0:
            return
        directory = Path(self.training_repo.value) / self.train_labels.value[0]
        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(directory.glob("**/*"), 10)
        ]
        self.test_labels.value = []

    @MLWidget.output.capture(clear_output=True)
    def update_test_file_list(self, *args):
        if len(self.test_labels.value) == 0:
            return
        directory = Path(self.testing_repo.value) / self.test_labels.value[0]
        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(directory.glob("**/*"), 10)
        ]
        self.train_labels.value = []

    @MLWidget.output.capture(clear_output=True)
    def display_img(self, args):
        for path in args["new"]:
            shape, img = img_handle(Path(path))
            if self.img_width.value == "":
                self.img_width.value = str(shape[0])
            if self.img_height.value == "":
                self.img_height.value = str(shape[1])
            display(
                img
            )  # TODO display next to each other with shape info as well

    def __init__(
        self,
        sname: str,
        *args,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        nclasses: int = -1,
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-4,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        gpuid: int = 0,
        layers: List[str] = [],
        template: Optional[str] = None,
        mirror: bool = False,
        rotate: bool = False,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: str = "SGD",
        noise_prob: float = 0.0,
        distort_prob: float = 0.0,
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        ctc: bool = False,
        timesteps: int = 32,
        unchanged_data: bool = False
    ) -> None:

        local_vars = locals()
        params = {
            # no access to eval(k) inside the comprehension
            k: (eval(k, local_vars), v)
            for k, v in get_type_hints(self.__init__).items()
            if k not in ["return", "sname"]
        }

        super().__init__(sname, params)

        self.train_labels = SelectMultiple(
            options=[], value=[], description="Training labels", disabled=False
        )

        self.test_labels = SelectMultiple(
            options=[], value=[], description="Testing labels", disabled=False
        )

        self.file_list = SelectMultiple(
            options=[],
            value=[],
            rows=10,
            description="File list",
            layout=Layout(height="200px"),
        )

        self.testing_repo.observe(self.update_label_list, names="value")
        self.training_repo.observe(self.update_label_list, names="value")

        self.train_labels.observe(self.update_train_file_list, names="value")
        self.test_labels.observe(self.update_test_file_list, names="value")
        self.file_list.observe(self.display_img, names="value")

        self._img_explorer = VBox(
            [
                HBox([HBox([self.train_labels, self.test_labels])]),
                self.file_list,
                self.output,
            ],
            layout=Layout(width="650px"),
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="900px"),
        )

        self.update_label_list(())

    @MLWidget.output.capture(clear_output=True)
    def run(self, *_) -> None:
        logging.info("Sending a classification task")
        send_dd(self)


class Segmentation(MLWidget):
    @MLWidget.output.capture(clear_output=True)
    def update_train_file_list(self, *args):
        # print (Path(self.training_repo.value).read_text().split('\n'))
        self.file_dict = {
            Path(x.split()[0]): Path(x.split()[1])
            for x in Path(self.training_repo.value).read_text().split("\n")
            if len(x.split()) >= 2
        }

        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(self.file_dict.keys(), 10)
        ]

    @MLWidget.output.capture(clear_output=True)
    def update_test_file_list(self, *args):
        # print (Path(self.training_repo.value).read_text().split('\n'))
        self.file_dict = {
            Path(x.split()[0]): Path(x.split()[1])
            for x in Path(self.testing_repo.value).read_text().split("\n")
            if len(x.split()) >= 2
        }

        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(self.file_dict.keys(), 10)
        ]

    def __init__(
        self,
        sname: str,
        *args,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        nclasses: int = -1,
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-3,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        gpuid: int = 0,
        layers: List[str] = [],
        template: Optional[str] = None,
        mirror: bool = True,
        rotate: bool = True,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: str = "SGD",
        noise_prob: float = 0.0,
        distort_prob: float = 0.0,
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        model_postfix: str = "",
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        ctc: bool = False,
        timesteps: int = 32,
        unchanged_data: bool = False
    ) -> None:

        local_vars = locals()
        params = {
            # no access to eval(k) inside the comprehension
            k: (eval(k, local_vars), v)
            for k, v in get_type_hints(self.__init__).items()
            if k not in ["return", "sname"]
        }

        super().__init__(sname, params)

        self.train_labels = Button(
            description=Path(self.training_repo.value).name
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name
        )

        self.file_list = SelectMultiple(
            options=[],
            value=[],
            rows=10,
            description="File list",
            layout=Layout(height="200px", width="auto"),
        )

        # self.testing_repo.observe(self.update_test_button, names="value")
        # self.training_repo.observe(self.update_train_button, names="value")

        self.train_labels.on_click(self.update_train_file_list)
        self.test_labels.on_click(self.update_test_file_list)

        self.file_list.observe(self.display_img, names="value")

        self._img_explorer = VBox(
            [
                HBox([HBox([self.train_labels, self.test_labels])]),
                self.file_list,
                self.output,
            ],
            layout=Layout(width="650px"),
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="900px"),
        )

        self.update_label_list(())

    @MLWidget.output.capture(clear_output=True)
    def display_img(self, args):
        for path in args["new"]:
            shape, img = img_handle(Path(path))
            if self.img_width.value == "":
                self.img_width.value = str(shape[0])
            if self.img_height.value == "":
                self.img_height.value = str(shape[1])
            display(
                img
            )  # TODO display next to each other with shape info as well
            _, img = img_handle(Path(path), self.file_dict[Path(path)])
            display(img)
            # display(Image(path))
            # integrate THIS : https://github.com/alx/react-bounding-box
            # (cv2.imread(self.file_dict[Path(path)].as_posix()))

    @MLWidget.output.capture(clear_output=True)
    def run(self, *_) -> None:
        logging.info("Sending a segmentation task")
        send_dd(self)


class Detection(MLWidget):
    @MLWidget.output.capture(clear_output=True)
    def display_img(self, args):
        for path in args["new"]:
            shape, img = img_handle(Path(path))
            if self.img_width.value == "":
                self.img_width.value = str(shape[0])
            if self.img_height.value == "":
                self.img_height.value = str(shape[1])
            _, img = img_handle(Path(path), bbox=self.file_dict[Path(path)])
            display(img)

    @MLWidget.output.capture(clear_output=True)
    def update_train_file_list(self, *args):
        # print (Path(self.training_repo.value).read_text().split('\n'))
        self.file_dict = {
            Path(x.split()[0]): Path(x.split()[1])
            for x in Path(self.training_repo.value).read_text().split("\n")
            if len(x.split()) >= 2
        }

        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(self.file_dict.keys(), 10)
        ]

    @MLWidget.output.capture(clear_output=True)
    def update_test_file_list(self, *args):
        # print (Path(self.training_repo.value).read_text().split('\n'))
        self.file_dict = {
            Path(x.split()[0]): Path(x.split()[1])
            for x in Path(self.testing_repo.value).read_text().split("\n")
            if len(x.split()) >= 2
        }

        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(self.file_dict.keys(), 10)
        ]

    def __init__(
        self,
        sname: str,
        *args,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        nclasses: int = -1,
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-4,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        gpuid: int = 0,
        layers: List[str] = [],
        template: Optional[str] = None,
        mirror: bool = False,
        rotate: bool = False,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: str = "SGD",
        noise_prob: float = 0.0,
        distort_prob: float = 0.0,
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        ctc: bool = False,
        timesteps: int = 32,
        unchanged_data: bool = False
    ) -> None:

        local_vars = locals()
        params = {
            # no access to eval(k) inside the comprehension
            k: (eval(k, local_vars), v)
            for k, v in get_type_hints(self.__init__).items()
            if k not in ["return", "sname"]
        }

        super().__init__(sname, params)

        self.train_labels = Button(
            description=Path(self.training_repo.value).name
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name
        )

        self.file_list = SelectMultiple(
            options=[],
            value=[],
            rows=10,
            description="File list",
            layout=Layout(height="200px", width="auto"),
        )

        self.train_labels.on_click(self.update_train_file_list)
        self.test_labels.on_click(self.update_test_file_list)

        self.file_list.observe(self.display_img, names="value")

        self._img_explorer = VBox(
            [
                HBox([HBox([self.train_labels, self.test_labels])]),
                self.file_list,
                self.output,
            ],
            layout=Layout(width="650px"),
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="900px"),
        )

        self.update_label_list(())

    @MLWidget.output.capture(clear_output=True)
    def run(self, *_) -> None:
        logging.info("Sending a classification task")
        send_dd(self)
