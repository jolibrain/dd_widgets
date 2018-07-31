# fmt: off

import json
import logging
import random
import time
from collections import OrderedDict
from enum import Enum
from heapq import nlargest
from inspect import signature
from pathlib import Path
from tempfile import mkstemp
from typing import (Any, Dict, Iterator, List, Optional, Tuple, TypeVar,
                    get_type_hints)

import matplotlib.pyplot as plt
from IPython.display import Image, display
from matplotlib import patches

import cv2
import pandas as pd
import requests
from core import ImageTrainerMixin
from ipywidgets import (HTML, Button, Checkbox, Dropdown, FloatText, HBox,
                        IntProgress, IntText, Label, Layout, Output,
                        SelectMultiple, Tab)
from ipywidgets import Text as TextWidget
from ipywidgets import VBox
from loghandler import OutputWidgetHandler

# fmt: on

# -- Logging --

fmt = "%(asctime)s:%(msecs)d - %(levelname)s"
fmt += " - {%(filename)s:%(lineno)d} %(message)s"

file_handler = logging.FileHandler("widgets.log")
widget_output_handler = OutputWidgetHandler()

logging.basicConfig(
    format=fmt,
    level=logging.DEBUG,
    datefmt="%m-%d %H:%M:%S",
    handlers=[file_handler, widget_output_handler],
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
        ax.imshow(data, alpha=.8)
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


class Solver(Enum):
    SGD = "SGD"
    ADAM = "ADAM"
    RMSPROP = "RMSPROP"
    AMSGRAD = "AMSGRAD"
    ADAGRAD = "ADAGRAD"
    ADADELTA = "ADADELTA"
    NESTEROV = "NESTEROV"


class SolverDropdown(Dropdown):
    def __init__(self, *args, **kwargs):
        Dropdown.__init__(
            self, *args, options=list(e.name for e in Solver), **kwargs
        )


# -- Core 'abstract' widget for many tasks


class MLWidget(object):

    _fields = {  # typing: Dict[str, str]
        "sname": "Model name",
        "training_repo": "Training directory",
        "testing_repo": "Testing directory",
    }

    _widget_type = {
        int: IntText,
        float: FloatText,
        bool: Checkbox,
        Solver: SolverDropdown,
    }

    # host: TextWidget
    # port: TextWidget

    def typing_info(self, local_vars: Dict[str, Any]):
        fun = self.__init__  # type: ignore
        typing_dict = get_type_hints(fun)
        for param in signature(fun).parameters.values():
            if param.name != "sname":
                yield (
                    param.name,
                    eval(param.name, local_vars),
                    typing_dict[param.name],
                )

    def __init__(self, sname: str, local_vars: Dict[str, Any], *args) -> None:

        # logger.addHandler(log_viewer(self.output),)

        self.sname = sname
        self.output = Output(layout=Layout(max_width="650px"))
        self.pbar = IntProgress(
            min=0,
            max=100,
            description="Progression:",
            layout=Layout(margin="18px"),
        )
        self.run_button = Button(description="Run")
        self.info_button = Button(description="Info")
        self.clear_button = Button(description="Clear")
        self.hardclear_button = Button(description="Hard Clear")

        self._widgets = [  # typing: List[Widget]
            HTML(
                value="<h2>{task} task: {sname}</h2>".format(
                    task=self.__class__.__name__, sname=self.sname
                )
            ),
            HBox([self.run_button, self.clear_button]),
            HBox([self.info_button, self.hardclear_button]),
        ]

        self.run_button.on_click(self.run)
        self.info_button.on_click(self.info)
        self.clear_button.on_click(self.clear)
        self.hardclear_button.on_click(self.hardclear)

        for name, value, type_hint in self.typing_info(local_vars):
            self._add_widget(name, value, type_hint)

        self._configuration = VBox(
            self._widgets, layout=Layout(min_width="250px")
        )

        self._tabs = Tab(layout=Layout(height=""))
        self._output = VBox([self.pbar, self._tabs])
        self._main_elt = HBox(
            [self._configuration, self._output], layout=Layout(width="1200px")
        )
        self._img_explorer = VBox(
            [self.output], layout=Layout(min_height="800px", width="590px")
        )

        self._tabs.children = [self._img_explorer, widget_output_handler.out]
        self._tabs.set_title(0, "Exploration")
        self._tabs.set_title(1, "Logs")

        self.file_list = SelectMultiple(
            options=[],
            value=[],
            rows=10,
            description="File list",
            layout=Layout(height="200px", width="560px"),
        )

    def _add_widget(self, name, value, type_hint):

        widget_type = self._widget_type.get(type_hint, None)

        if widget_type is None:
            setattr(
                self,
                name,
                TextWidget(  # Widget type by default then convert to str
                    value="" if value is None else str(value),
                    layout=Layout(min_width="20ex", margin="-2px 2px 4px 2px"),
                ),
            )
            self._widgets.append(
                VBox(
                    [
                        Label(self._fields.get(name, name) + ":"),
                        getattr(self, name),
                    ]
                )
            )
        else:
            setattr(
                self,
                name,
                widget_type(
                    value=type_hint() if value is None else (value),
                    layout=Layout(width="100px", margin="4px 2px 4px 2px"),
                ),
            )

            self._widgets.append(
                HBox(
                    [
                        Label(
                            self._fields.get(name, name),
                            layout=Layout(min_width="180px"),
                        ),
                        getattr(self, name),
                    ],
                    layout=Layout(margin="4px 2px 4px 2px"),
                )
            )

    def _ipython_display_(self):
        self._main_elt._ipython_display_()

    def clear(self, *_):
        self.output.clear_output()
        with self.output:
            request = "http://{host}:{port}/{path}/services/{sname}?clear=full".format(
                host=self.host.value,
                port=self.port.value,
                path=self.path.value,
                sname=self.sname,
            )
            c = requests.delete(request)
            logging.info(
                "Clearing (full) service {sname}: {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )

            print(json.dumps(c.json(), indent=2))
            return c.json()

    def hardclear(self, *_):
        # The basic version
        self.output.clear_output()
        with self.output:
            MLWidget.create_service(self)
            self.clear()

    def create_service(self, *_):
        with self.output:
            host = self.host.value
            port = self.port.value

            body = OrderedDict(
                [
                    ("mllib", "caffe"),
                    ("description", self.sname),
                    ("type", "supervised"),
                    (
                        "parameters",
                        {
                            "mllib": {"nclasses": 42},  # why not?
                            "input": {"connector": "csv"},
                        },
                    ),
                    (
                        "model",
                        {
                            "repository": self.model_repo.value,
                            "create_repository": True,
                            # "templates": "../templates/caffe/"
                        },
                    ),
                ]
            )

            logging.info(
                "Creating service '{sname}':\n {body}".format(
                    sname=self.sname, body=json.dumps(body, indent=2)
                )
            )
            c = requests.put(
                "http://{host}:{port}/{path}/services/{sname}".format(
                    host=host, port=port, path=self.path.value, sname=self.sname
                ),
                json.dumps(body),
            )

            if c.json()["status"]["code"] != 201:
                logging.warning(
                    "Reply from creating service '{sname}': {json}".format(
                        sname=self.sname, json=json.dumps(c.json(), indent=2)
                    )
                )
            else:
                logging.info(
                    "Reply from creating service '{sname}': {json}".format(
                        sname=self.sname, json=json.dumps(c.json(), indent=2)
                    )
                )

            print(json.dumps(c.json(), indent=2))

            return c.json()

    def run(self, *_):
        logging.info("Entering run method")
        self.output.clear_output()

        with self.output:
            host = self.host.value
            port = self.port.value
            body = self._create_service_body()

            logging.info(
                "Sending request http://{host}:{port}/{path}/services/{sname}".format(
                    host=host, port=port, path=self.path.value, sname=self.sname
                )
            )
            c = requests.get(
                "http://{host}:{port}/{path}/services/{sname}".format(
                    host=host, port=port, path=self.path.value, sname=self.sname
                )
            )
            logging.info(
                "Current state of service '{sname}': {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )
            if c.json()["status"]["msg"] != "NotFound":
                self.clear()
                logging.warning(
                    (
                        "Since service '{sname}' was still there, "
                        "it has been fully cleared: {json}"
                    ).format(
                        sname=self.sname, json=json.dumps(c.json(), indent=2)
                    )
                )

            logging.info(
                "Creating service '{sname}':\n {body}".format(
                    sname=self.sname, body=json.dumps(body, indent=2)
                )
            )
            c = requests.put(
                "http://{host}:{port}/{path}/services/{sname}".format(
                    host=host, port=port, path=self.path.value, sname=self.sname
                ),
                json.dumps(body),
            )

            if c.json()["status"]["code"] != 201:
                logging.warning(
                    "Reply from creating service '{sname}': {json}".format(
                        sname=self.sname, json=json.dumps(c.json(), indent=2)
                    )
                )
                return
            else:
                logging.info(
                    "Reply from creating service '{sname}': {json}".format(
                        sname=self.sname, json=json.dumps(c.json(), indent=2)
                    )
                )

            body = self._train_body()

            logging.info(
                "Start training phase: {body}".format(
                    body=json.dumps(body, indent=2)
                )
            )
            c = requests.post(
                "http://{host}:{port}/{path}/train".format(
                    host=host, port=port, path=self.path.value
                ),
                json.dumps(body),
            )
            logging.info(
                "Reply from training service '{sname}': {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )

            print(json.dumps(c.json(), indent=2))

            self.value = self.iterations.value
            self.pbar.bar_style = "info"
            self.pbar.max = self.iterations.value

            while True:
                info = self.info(print_output=False)
                self.pbar.bar_style = ""

                status = info["head"]["status"]

                if status == "finished":
                    self.pbar.value = self.iterations.value
                    self.pbar.bar_style = "success"
                    break

                self.pbar.value = info["body"]["measure"].get("iteration", 0)

                time.sleep(1)

    def info(self, print_output=True):
        with self.output:
            # TODO job number
            request = (
                "http://{host}:{port}/{path}/train?service={sname}&"
                "job=1&timeout=10".format(
                    host=self.host.value,
                    port=self.port.value,
                    path=self.path.value,
                    sname=self.sname,
                )
            )
            c = requests.get(request)
            logging.info(
                "Getting info for service {sname}: {json}".format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )
            if print_output:
                print(json.dumps(c.json(), indent=2))
            return c.json()

    def update_label_list(self, _):
        with self.output:
            if self.training_repo.value != "":
                self.train_labels.options = tuple(
                    sorted(
                        f.stem for f in Path(self.training_repo.value).glob("*")
                    )
                )
            if self.testing_repo.value != "":
                self.test_labels.options = tuple(
                    sorted(
                        f.stem for f in Path(self.testing_repo.value).glob("*")
                    )
                )

            self.train_labels.rows = min(10, len(self.train_labels.options))
            self.test_labels.rows = min(10, len(self.test_labels.options))
            if self.nclasses.value == -1:
                self.nclasses.value = str(len(self.train_labels.options))


class Classification(MLWidget, ImageTrainerMixin):
    ctc = False
    def update_train_file_list(self, *args):
        with self.output:
            if len(self.train_labels.value) == 0:
                return
            directory = (
                Path(self.training_repo.value) / self.train_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.test_labels.value = []

    def update_test_file_list(self, *args):
        with self.output:
            if len(self.test_labels.value) == 0:
                return
            directory = (
                Path(self.testing_repo.value) / self.test_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.train_labels.value = []

    def display_img(self, args):
        self.output.clear_output()
        with self.output:
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
        *,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        nclasses: int = -1,
        description: str = "classification service",
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
        solver_type: Solver = "SGD",
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
        timesteps: int = 32,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = SelectMultiple(
            options=[], value=[], description="Training labels", disabled=False
        )

        self.test_labels = SelectMultiple(
            options=[], value=[], description="Testing labels", disabled=False
        )

        self.testing_repo.observe(self.update_label_list, names="value")
        self.training_repo.observe(self.update_label_list, names="value")

        self.train_labels.observe(self.update_train_file_list, names="value")
        self.test_labels.observe(self.update_test_file_list, names="value")
        self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        self.update_label_list(())


class Segmentation(MLWidget, ImageTrainerMixin):
    def update_train_file_list(self, *args):
        with self.output:
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

    def update_test_file_list(self, *args):
        with self.output:
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
        *,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        nclasses: int = -1,
        description: str = "Segmentation service",
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
        solver_type: Solver = "SGD",
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
        timesteps: int = 32,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = "",
        loss: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = Button(
            description=Path(self.training_repo.value).name
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name
        )

        # self.testing_repo.observe(self.update_test_button, names="value")
        # self.training_repo.observe(self.update_train_button, names="value")

        self.train_labels.on_click(self.update_train_file_list)
        self.test_labels.on_click(self.update_test_file_list)

        self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        self.update_label_list(())

    def display_img(self, args):
        self.output.clear_output()
        with self.output:
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


class Detection(MLWidget, ImageTrainerMixin):
    ctc = False
    def display_img(self, args):
        self.output.clear_output()
        with self.output:
            for path in args["new"]:
                shape, img = img_handle(Path(path))
                if self.img_width.value == "":
                    self.img_width.value = str(shape[0])
                if self.img_height.value == "":
                    self.img_height.value = str(shape[1])
                _, img = img_handle(Path(path), bbox=self.file_dict[Path(path)])
                display(img)

    def update_train_file_list(self, *args):
        with self.output:
            self.file_dict = {
                Path(x.split()[0]): Path(x.split()[1])
                for x in Path(self.training_repo.value).read_text().split("\n")
                if len(x.split()) >= 2
            }

            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(self.file_dict.keys(), 10)
            ]

    def update_test_file_list(self, *args):
        with self.output:
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
        *,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        nclasses: int = -1,
        description: str = "Detection service",
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        db_width: int = 0,
        db_height: int = 0,
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
        solver_type: Solver = "SGD",
        noise_prob: float = 0.001,
        distort_prob: float = 0.5,
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        timesteps: int = 32,
        unchanged_data: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = Button(
            description=Path(self.training_repo.value).name
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name
        )

        self.train_labels.on_click(self.update_train_file_list)
        self.test_labels.on_click(self.update_test_file_list)

        self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        self.update_label_list(())


class CSV(MLWidget):
    def __init__(
        self,
        sname: str,
        *,
        training_repo: Path = None,
        testing_repo: Path = None,
        description: str = "CSV service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        tsplit: float = 0.01,
        base_lr: float = 0.01,
        iterations: int = 100,
        test_interval: int = 1000,
        step_size: int = 0,
        template: Optional[str] = None,
        layers: List[int] = [],
        activation: str = "relu",
        db: bool = False,
        dropout: float = .2,
        destroy: bool = False,
        resume: bool = False,
        finetune: bool = False,
        weights: Optional[Path] = None,
        nclasses: int = 2,
        batch_size: int = 128,
        test_batch_size: int = 16,
        gpuid: int = 0,
        mllib: str = "caffe",
        lregression: bool = False,
        scale: bool = False,
        csv_id: str = "",
        csv_separator: str = ",",
        csv_ignore: List[str] = [],
        csv_label: str = "",
        csv_label_offset: int = -1,
        csv_categoricals: List[str] = [],
        scale_pos_weight: float = 1.0,
        shuffle: bool = True,
        solver_type: Solver = "AMSGRAD",
        autoencoder: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self._displays = HTML(
            value=pd.read_csv(training_repo).sample(5)._repr_html_()
        )
        self._img_explorer.children = [self._displays, self.output]

    def _create_service_body(self):
        body = OrderedDict(
            [
                ("mllib", "caffe"),
                ("description", self.sname),
                ("type", "supervised"),
                (
                    "parameters",
                    {
                        "input": {
                            "connector": "csv",
                            "labels": self.csv_label.value,
                            "db": self.db.value,
                        },
                        "mllib": {
                            "template": self.template.value,
                            "nclasses": self.nclasses.value,
                            "activation": self.activation.value,
                            "db": self.db.value,
                            "template": self.template.value,
                            "layers": eval(self.layers.value),
                            "autoencoder": self.autoencoder.value,
                        },
                    },
                ),
                (
                    "model",
                    {
                        "templates": "../templates/caffe/",
                        "repository": self.model_repo.value,
                        "create_repository": True,
                    },
                ),
            ]
        )

        if self.lregression.value:
            body["parameters"]["mllib"]["template"] = "lregression"
            del body["parameters"]["mllib"]["layes"]
        else:
            body["parameters"]["mllib"]["dropout"] = self.dropout.value

        if self.mllib.value == "xgboost":
            body["parameters"]["mllib"]["db"] = False

        if self.finetune.value:
            body["parameters"]["mllib"]["finetuning"] = True
            body["parameters"]["mllib"]["weights"] = self.weights.value

        return body

    def _train_body(self):
        body = OrderedDict(
            [
                ("service", self.sname),
                ("async", True),
                (
                    "parameters",
                    {
                        "mllib": {
                            "gpu": True,
                            "gpuid": self.gpuid.value,
                            "resume": self.resume.value,
                            "solver": {
                                "iterations": self.iterations.value,
                                "iter_size": 1,
                                "test_interval": self.test_interval.value,
                                "test_initialization": False,
                                "base_lr": self.base_lr.value,
                                "solver_type": self.solver_type.value,
                            },
                            "net": {
                                "batch_size": self.batch_size.value,
                                "test_batch_size": self.test_batch_size.value,
                            },
                        },
                        "input": {
                            "label_offset": self.csv_label_offset.value,
                            "label": self.csv_label.value,
                            "id": self.csv_id.value,
                            "separator": self.csv_separator.value,
                            "shuffle": self.shuffle.value,
                            "test_split": self.tsplit.value,
                            "scale": self.scale.value,
                            "db": self.db.value,
                            "ignore": eval(self.csv_ignore.value),
                            "categoricals": eval(self.csv_categoricals.value),
                            "autoencoder": self.autoencoder.value,
                        },
                        "output": {
                            "measure": ["cmdiag", "cmfull", "mcll", "f1"]
                        },
                    },
                ),
                ("data", [self.training_repo.value, self.testing_repo.value]),
            ]
        )

        if self.nclasses.value == 2:
            body["parameters"]["output"]["measure"].append("auc")

        if self.autoencoder.value:
            body["parameters"]["output"]["measure"] = ["eucll"]

        return body


class Text(MLWidget):
    def __init__(
        self,
        sname: str,
        *,
        training_repo: Path,
        testing_repo: Optional[Path] = None,
        description: str = "Text service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        db: bool = False,
        nclasses: int = -1,
        layers: List[str] = [],
        gpuid: int = 0,
        iterations: int = 25000,
        test_interval: int = 1000,
        base_lr: float = 0.001,
        solver_type: Solver = "SGD",
        batch_size: int = 128,
        shuffle: bool = True,
        tsplit: float = 0.2,
        min_count: int = 10,
        min_word_length: int = 5,
        count: bool = False,
        tfidf: bool = False,
        sentences: bool = False,
        characters: bool = False,
        sequence: int = -1,
        read_forward: bool = True,
        alphabet: str = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:’\“/\_@#$%^&*~`+-=<>()[]{}",
        sparse: bool = False,
        template: str = "mlp",
        activation: str = "relu",
        embedding: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = SelectMultiple(
            options=[], value=[], description="Training labels", disabled=False
        )

        self.test_labels = SelectMultiple(
            options=[], value=[], description="Testing labels", disabled=False
        )

        # self.testing_repo.observe(self.update_label_list, names="value")
        self.training_repo.observe(self.update_label_list, names="value")

        self.train_labels.observe(self.update_train_file_list, names="value")
        self.test_labels.observe(self.update_test_file_list, names="value")
        self.file_list.observe(self.display_text, names="value")

        self.update_label_list(())

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        if self.characters:
            self.db = True

    def display_text(self, args):
        self.output.clear_output()
        with self.output:
            for path in args["new"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, x in enumerate(fh.readlines()):
                        if i == 20:
                            break
                        print(x.strip())

    def update_train_file_list(self, *args):
        with self.output:
            if len(self.train_labels.value) == 0:
                return
            directory = (
                Path(self.training_repo.value) / self.train_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.test_labels.value = []

    def update_test_file_list(self, *args):
        with self.output:
            if len(self.test_labels.value) == 0:
                return
            directory = (
                Path(self.testing_repo.value) / self.test_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.train_labels.value = []

    def _create_service_body(self):

        body = OrderedDict(
            [
                ("mllib", "caffe"),
                ("description", "newsgroup classification service"),
                ("type", "supervised"),
                (
                    "parameters",
                    {
                        "input": {
                            "connector": "txt",
                            "characters": self.characters.value,
                            "sequence": self.sequence.value,
                            "read_forward": self.read_forward.value,
                            "alphabet": self.alphabet.value,
                            "sparse": self.sparse.value,
                            "embedding": self.embedding.value,
                        },
                        "mllib": {
                            "template": self.template.value,
                            "nclasses": self.nclasses.value,
                            "layers": eval(self.layers.value),
                            "activation": self.activation.value,
                        },
                    },
                ),
                (
                    "model",
                    {
                        "templates": "../templates/caffe/",
                        "repository": self.model_repo.value,
                        "create_repository": True,
                    },
                ),
            ]
        )
        return body

    def _train_body(self):
        body = OrderedDict(
            [
                ("service", self.sname),
                ("async", True),
                (
                    "parameters",
                    {
                        "mllib": {
                            "gpu": True,
                            "gpuid": self.gpuid.value,
                            "solver": {
                                "iterations": self.iterations.value,
                                "test_interval": self.test_interval.value,
                                "test_initialization": False,
                                "base_lr": self.base_lr.value,
                                "solver_type": self.solver_type.value,
                            },
                            "net": {"batch_size": self.batch_size.value},
                        },
                        "input": {
                            "shuffle": self.shuffle.value,
                            "test_split": self.tsplit.value,
                            "min_count": self.min_count.value,
                            "min_word_length": self.min_word_length.value,
                            "count": self.count.value,
                            "tfidf": self.tfidf.value,
                            "sentences": self.sentences.value,
                            "characters": self.characters.value,
                            "sequence": self.sequence.value,
                            "read_forward": self.read_forward.value,
                            "alphabet": self.alphabet.value,
                            "embedding": self.embedding.value,
                            "db": self.db.value,
                        },
                        "output": {"measure": ["mcll", "f1", "cmdiag"]},
                    },
                ),
                ("data", [self.training_repo.value]),
            ]
        )

        return body

class OCR(MLWidget, ImageTrainerMixin):
    
    ctc = True

    def update_train_file_list(self, *args):
        with self.output:
            # print (Path(self.training_repo.value).read_text().split('\n'))
            self.file_dict = {
                Path(x.split()[0]): x.split()[1:]
                for x in Path(self.training_repo.value).read_text().split("\n")
                if len(x.split()) >= 2
            }

            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(self.file_dict.keys(), 10)
            ]

    def update_test_file_list(self, *args):
        with self.output:
            # print (Path(self.training_repo.value).read_text().split('\n'))
            self.file_dict = {
                Path(x.split()[0]): x.split()[1:]
                for x in Path(self.testing_repo.value).read_text().split("\n")
                if len(x.split()) >= 2
            }

            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(self.file_dict.keys(), 10)
            ]
            
    def display_img(self, args):
        self.output.clear_output()
        with self.output:
            for path in args["new"]:
                shape, img = img_handle(Path(path))
                if self.img_width.value == "":
                    self.img_width.value = str(shape[0])
                if self.img_height.value == "":
                    self.img_height.value = str(shape[1])
                display(
                    img
                )  # TODO display next to each other with shape info as well
                print(" ".join(o.file_dict[Path(path)]))

    def __init__(
        self,
        sname: str,
        *,  # unnamed parameters are forbidden
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        nclasses: int = -1,
        description: str = "OCR service",
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
        solver_type: Solver = "SGD",
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
        timesteps: int = 32,
        unchanged_data: bool = False,
        target_repository: str = "",
        align: bool = False
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = Button(
            description=Path(self.training_repo.value).name
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name
        )

        # self.testing_repo.observe(self.update_test_button, names="value")
        # self.training_repo.observe(self.update_train_button, names="value")

        self.train_labels.on_click(self.update_train_file_list)
        self.test_labels.on_click(self.update_test_file_list)

        
        self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        #self.update_label_list(())
