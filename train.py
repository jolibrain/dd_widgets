
import random
from heapq import nlargest
from pathlib import Path
from tempfile import mkstemp
from typing import Iterator, List, Optional, Tuple, TypeVar, get_type_hints

import matplotlib.pyplot as plt
from IPython.display import Image, display

import cv2
from ipywidgets import (
    Button,
    Checkbox,
    FloatText,
    HBox,
    IntText,
    Layout,
    Output,
    SelectMultiple,
    Text,
    VBox,
)


def img_handle(path: Path) -> Tuple[Tuple[int, ...], Image]:
    data = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    _, fname = mkstemp(suffix=".png")
    fig, ax = plt.subplots()
    ax.imshow(data)
    fig.savefig(fname)
    plt.close(fig)
    return data.shape, Image(fname)


Elt = TypeVar("Elt")


def sample_from_iterable(it: Iterator[Elt], k: int) -> Iterator[Elt]:
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


class Classification(object):

    _fields = {
        "sname": "Model name",
        "training_repo": "Training directory",
        "testing_repo": "Testing directory",
    }

    _widget_type = {int: IntText, float: FloatText, bool: Checkbox}

    output = Output()

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

    @output.capture(clear_output=True)
    def update_train_file_list(self, *args):
        if len(self.train_labels.value) == 0:
            return
        directory = Path(self.training_repo.value) / self.train_labels.value[0]
        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(directory.glob("**/*"), 10)
        ]
        self.test_labels.value = []

    @output.capture(clear_output=True)
    def update_test_file_list(self, *args):
        if len(self.test_labels.value) == 0:
            return
        directory = Path(self.testing_repo.value) / self.test_labels.value[0]
        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(directory.glob("**/*"), 10)
        ]
        self.train_labels.value = []

    @output.capture(clear_output=True)
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

    def _add_widget(self, name, value, type_hint):

        widget_type = self._widget_type.get(type_hint, None)

        if widget_type is None:
            setattr(
                self,
                name,
                Text(
                    value="" if value is None else str(value),
                    description=self._fields.get(name, name)
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
        segmentation: bool = False,
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

        elts = {
            k: v
            for k, v in get_type_hints(self.__init__).items()
            if k != "return"
        }

        self._widgets = []

        for elt, type_hint in elts.items():
            self._add_widget(elt, eval(elt), type_hint)

        self.run_button = Button(description="Run")
        self._widgets.append(self.run_button)

        self._configuration = VBox(self._widgets)

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
            layout=Layout(width="95%", height="200px"),
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
            ]
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="100%"),
        )

        self.run_button.on_click(self.run)
        self.update_label_list(())

    @output.capture(clear_output=True)
    def run(self, *_) -> None:
        # ça me dépanne pour le moment
        print(get_type_hints(self.__init__))

    def _ipython_display_(self):
        self._main_elt._ipython_display_()


class Segmentation(Classification):

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
        segmentation: bool = False,
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

        elts = {
            k: v
            for k, v in get_type_hints(self.__init__).items()
            if k != "return"
        }

        self._widgets = []

        for elt, type_hint in elts.items():
            self._add_widget(elt, eval(elt), type_hint)

        self.run_button = Button(description="Run")
        self._widgets.append(self.run_button)

        self._configuration = VBox(self._widgets)

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
            layout=Layout(width="95%", height="200px"),
        )

        #self.testing_repo.observe(self.update_test_button, names="value")
        #self.training_repo.observe(self.update_train_button, names="value")

        self.train_labels.on_click(self.update_train_file_list)
        #self.test_labels.on_click(self.update_test_file_list)

        self.file_list.observe(self.display_img, names="value")

        self._img_explorer = VBox(
            [
                HBox([HBox([self.train_labels, self.test_labels])]),
                self.file_list,
                self.output,
            ]
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="100%"),
        )

        self.run_button.on_click(self.run)
        self.update_label_list(())

    @Classification.output.capture(clear_output=True)
    def update_train_file_list(self, *args):
        #print (Path(self.training_repo.value).read_text().split('\n'))
        self.file_dict = {Path(x.split()[0]): Path(x.split()[1])
                     for x in Path(self.training_repo.value).read_text().split('\n')
                   if len(x.split()) >= 2}

        self.file_list.options = [
            fh.as_posix()
            for fh in sample_from_iterable(self.file_dict.keys(), 10)
        ]

    @Classification.output.capture(clear_output=True)
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
            display(Image(Path(path).read_bytes()))
            # integrate THIS : https://github.com/alx/react-bounding-box
            print(cv2.imread(self.file_dict[Path(path)].as_posix()))
