# fmt: off

import json
import logging
import os
import random
import shutil
from heapq import nlargest
from pathlib import Path
from tempfile import mkstemp
from typing import (Any, Dict, Iterator, List, Optional, Tuple, TypeVar,
                    get_type_hints)

import matplotlib.pyplot as plt
from IPython.display import Image, display

import cv2
import requests
from ipywidgets import (Button, Checkbox, FloatText, HBox, IntText, Layout,
                        Output, SelectMultiple, Text, VBox, Widget, HTML)

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

from matplotlib import patches 


def img_handle(
    path: Path, segmentation: Optional[Path] = None, bbox: Optional[Path] = None
) -> Tuple[Tuple[int, ...], Image]:
    data = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    _, fname = mkstemp(suffix=".png")
    fig, ax = plt.subplots()
    ax.imshow(data)
    if segmentation is not None:
        data = cv2.imread(path2.as_posix(), cv2.IMREAD_UNCHANGED)
        ax.imshow(data, alpha=.2)
    if bbox is not None:
        with bbox.open('r') as fh:
            for line in fh.readlines():
                tag, xmin, ymin, xmax, ymax = (int(x) for x in line.strip().split())
                rect = patches.Rectangle((xmin,ymin),xmax-xmin,ymax-ymin,
                                         linewidth=2,edgecolor='blue',facecolor='none')
                ax.add_patch(rect)
              
    fig.savefig(fname)
    plt.close(fig)
    return data.shape, Image(fname)


Elt = TypeVar("Elt")


def sample_from_iterable(it: Iterator[Elt], k: int) -> Iterator[Elt]:
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


# -- Core 'abstract' widget for many tasks

class MLWidget(object):

    _fields = { # typing: Dict[str, str]
        "sname": "Model name",
        "training_repo": "Training directory",
        "testing_repo": "Testing directory",
    }

    _widget_type = {int: IntText, float: FloatText, bool: Checkbox}

    output = Output(layout=Layout(max_width='650px')) # typing: Output
    #host: Text
    #port: Text
        
    def __init__(self, sname: str, params: Dict[str, Tuple[Any, type]], *args):
        self.sname = sname
        
        self.run_button = Button(description="Run")
        self.info_button = Button(description="Info")
        self.clear_button = Button(description="Clear")
        
        self._widgets = [ # typing: List[Widget]
            HTML(value="<h2>{task} task: {sname}</h2>".format(
                task=self.__class__.__name__,
                sname=self.sname
            )
                ),
            self.run_button,
            self.info_button,
            self.clear_button
        ]

        self.run_button.on_click(self.run)
        self.info_button.on_click(self.info)
        self.clear_button.on_click(self.clear)
        
        for name, (value, type_hint) in params.items():
            self._add_widget(name, value, type_hint)

        self._configuration = VBox(self._widgets,
                                   layout=Layout(min_width='250px'))

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
        logging.info("Clearing (full) service {sname}: {json}".format(
            sname=self.sname,
            json=json.dumps(c.json(), indent=2)
        ))
        print(json.dumps(c.json(), indent=2))

    @output.capture(clear_output=True)
    def info(self, *_):
        # TODO job number
        request = (
            "http://{host}:{port}/train?service={sname}&job=1&timeout=10".format(
                host=self.host.value,
                port=self.port.value,
                sname=self.sname,
            )
        )
        c = requests.get(request)
        logging.info("Getting info for service {sname}: {json}".format(
            sname=self.sname,
            json=json.dumps(c.json(), indent=2)
        ))
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
            layout=Layout(width="650px")
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="900px"),
        )

        
        self.update_label_list(())

    @MLWidget.output.capture(clear_output=True)
    def run(self, *_) -> None:
        width = int(self.img_width.value)
        height = int(self.img_height.value)
        crop_size = int(self.crop_size.value)

        nclasses = int(self.nclasses.value)
        if nclasses == -1:
            nclasses = len(os.walk(self.training_repo.value).next()[1])

        host = self.host.value
        port = self.port.value
        description = "imagenet classifier"
        mllib = "caffe"

        model = {
            "templates": "../templates/caffe/",
            "repository": self.model_repo.value,
            "create_repository": True,
        }

        if self.weights.value:
            if not Path(self.model_repo.value).is_dir():
                logging.warn(
                    "Creating repository directory: {}".format(
                        self.model_repo.value
                    )
                )
                Path(self.model_repo.value).mkdir(parents=True)
                # change permission if dede is not run by current user
                Path(self.model_repo.value).chmod(0o777)

            shutil.copy(self.weights.value, self.model_repo.value + "/")

        parameters_input = {
            "connector": "image",
            "width": width,
            "height": height,
            "bw": self.bw.value,
            "db": True,
        }

        if self.multi_label.value:
            parameters_input["multi_label"] = True
            parameters_input["db"] = False
        if self.ctc.value:
            parameters_input["ctc"] = True
        if not self.finetune.value:
            if self.template.value:
                parameters_mllib = {
                    "template": self.template.value,
                    "nclasses": nclasses,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "layers": eval(self.layers.value),  # List of strings
                    "db": True,
                }
            else:
                parameters_mllib = {
                    "nclasses": nclasses,
                    "mirror": self.mirror.value,
                    "rotate": self.rotate.value,
                }
        else:
            if self.template.value:
                parameters_mllib = {
                    "template": self.template.value,
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": weights,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                }
            else:
                parameters_mllib = {
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": weights,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                }
        if self.multi_label.value:
            parameters_mllib["db"] = False

        if crop_size > 0:
            parameters_mllib["crop_size"] = crop_size
        if self.noise_prob.value > 0.0:
            parameters_mllib["noise"] = {
                "all_effects": True,
                "prob": self.noise_prob.value,
            }
        if self.distort_prob.value > 0.0:
            parameters_mllib["distort"] = {
                "all_effects": True,
                "prob": self.distort_prob.value,
            }
        parameters_mllib["gpu"] = True
        parameters_mllib["gpuid"] = self.gpuid.value
        if self.regression.value:
            parameters_mllib["regression"] = True

        parameters_output = {}
        # print (parameters_input)
        # print (parameters_mllib)
        # pserv = dd.put_service(self.sname.value,model,description,mllib,
        #                       parameters_input,parameters_mllib,parameters_output)

        body = { # typing: Dict[str, Any]
            "description": description,
            "mllib": mllib,
            "type": "supervised",
            "parameters": {
                "input": parameters_input,
                "mllib": parameters_mllib,
                "output": parameters_output,
            },
            "model": model,
        }

        c = requests.get(
            "http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            )
        )
        logging.info(
            "Current state of service '{sname}':\n  {json}".format(
                sname=self.sname, json=c.json()
            )
        )
        # useful for the clear() method
        if c.json()["status"]["msg"] != "NotFound":
            self.clear()
            logging.warn(
                (
                    "Since service '{sname}' was still there, "
                    "it has been fully cleared: {json}"
                ).format(sname=self.sname, json=c.json())
            )

        logging.info(
            "Creating service '{sname}':\n {body}".format(
                sname=self.sname, body=body
            )
        )
        c = requests.put(
            "http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            ),
            json.dumps(body),
        )
        logging.info(
            "Reply from creating service '{sname}': {json}".format(
                sname=self.sname, json=c.json()
            )
        )

        train_data = [self.training_repo.value]
        parameters_input = {
            "test_split": self.tsplit.value,
            "shuffle": True,
            "db": True,
        }
        if self.testing_repo.value != "":
            train_data.append(self.testing_repo.value)
            parameters_input = {"shuffle": True}

        if self.multi_label.value:
            parameters_input["db"] = False
        parameters_mllib = {
            "gpu": True,
            "gpuid": self.gpuid.value,
            "resume": self.resume.value,
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
            "solver": {
                "test_initialization": self.test_init.value,
                "iterations": self.iterations.value,
                "test_interval": self.test_interval.value,
                "snapshot": self.snapshot_interval.value,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
                "iter_size": self.iter_size.value,
            },
        }
        ##TODO: lr policy as arguments
        # 'lr_policy':'step','stepsize':2000,'gamma':0.1,'snapshot':4000,'base_lr':args.base_lr,'solver_type':'SGD'}}
        if self.rand_skip.value > 0 and self.resume.value:
            parameters_mllib["solver"]["rand_skip"] = self.rand_skip.value
        if self.class_weights.value:
            parameters_mllib["class_weights"] = eval(self.class_weights.value)
        if self.ignore_label.value >= 0:
            parameters_mllib["ignore_label"] = self.ignore_label.value
        if self.timesteps.value:
            parameters_mllib["timesteps"] = self.timesteps.value

        if self.multi_label.value and self.regression.value:
            parameters_output = {
                "measure": [
                    "kl",
                    "js",
                    "was",
                    "ks",
                    "dc",
                    "r2",
                    "deltas",
                    "eucll",
                ]
            }
        elif self.ctc.value:
            parameters_output = {"measure": ["acc"]}
        else:
            parameters_output = {"measure": ["mcll", "f1", "acc-5"]}

        body = {
            "service": self.sname,
            "async": True,
            "parameters": {
                "input": parameters_input,
                "mllib": parameters_mllib,
                "output": parameters_output,
            },
            "data": train_data,
        }

        logging.info("Start training phase: {body}".format(body=body))
        c = requests.post(
            "http://{host}:{port}/train".format(host=host, port=port),
            json.dumps(body),
        )
        logging.info(
            "Reply from training service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        print(json.dumps(c.json(), indent=2))




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
            layout=Layout(width="650px")
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

        logging.info("Running segmentation run")

        width = int(self.img_width.value)
        height = int(self.img_height.value)
        crop_size = int(self.crop_size.value)

        nclasses = int(self.nclasses.value)
        if nclasses == -1:
            import os

            logging.info("walking training repo")
            nclasses = len(os.walk(self.training_repo.value).next()[1])

        logging.info("{} classes".format(nclasses))
        host = self.host.value
        port = self.port.value
        description = "imagenet classifier"
        mllib = "caffe"

        model = {
            "templates": "../templates/caffe/",
            "repository": self.model_repo.value,
            "create_repository": True,
        }

        if self.weights.value:
            if not Path(self.model_repo.value).is_dir():
                logging.warn(
                    "Creating repository directory: {}".format(
                        self.model_repo.value
                    )
                )
                Path(self.model_repo.value).mkdir(parents=True)
                # change permission if dede is not run by current user
                Path(self.model_repo.value).chmod(0o777)

            shutil.copy(self.weights.value, self.model_repo.value + "/")

        parameters_input = {
            "connector": "image",
            "width": width,
            "height": height,
            "bw": self.bw.value,
            "db": True,
            "segmentation": True,
        }

        if self.multi_label.value:
            parameters_input["multi_label"] = True
            parameters_input["db"] = False
        if self.ctc.value:
            parameters_input["ctc"] = True

        logging.info("Parameters input: {}".format(parameters_input))

        if not self.finetune.value:
            if self.template.value:
                parameters_mllib = {
                    "template": self.template.value,
                    "nclasses": nclasses,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "layers": eval(self.layers.value),  # list of strings
                    "db": True,
                }
            else:
                parameters_mllib = {
                    "nclasses": nclasses,
                    "mirror": self.mirror.value,
                    "rotate": self.rotate.value,
                }
        else:
            if self.template.value:
                parameters_mllib = {
                    "template": self.template.value,
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": weights,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                }
            else:
                parameters_mllib = {
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": weights,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                }
        if self.multi_label.value:
            parameters_mllib["db"] = False

        if crop_size > 0:
            parameters_mllib["crop_size"] = crop_size
        if self.noise_prob.value > 0.0:
            parameters_mllib["noise"] = {
                "all_effects": True,
                "prob": self.noise_prob.value,
            }
        if self.distort_prob.value > 0.0:
            parameters_mllib["distort"] = {
                "all_effects": True,
                "prob": self.distort_prob.value,
            }
        parameters_mllib["gpu"] = True
        parameters_mllib["gpuid"] = self.gpuid.value
        if self.regression.value:
            parameters_mllib["regression"] = True

        logging.info("Parameters mllib: {}".format(parameters_input))

        parameters_output = {}
        # print (parameters_input)
        # print (parameters_mllib)
        # pserv = dd.put_service(self.sname.value,model,description,mllib,
        #                       parameters_input,parameters_mllib,parameters_output)

        body = {
            "description": description,
            "mllib": mllib,
            "type": "supervised",
            "parameters": {
                "input": parameters_input,
                "mllib": parameters_mllib,
                "output": parameters_output,
            },
            "model": model,
        }

        logging.info(
            "Sending request http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            )
        )
        c = requests.get(
            "http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            )
        )
        logging.info(
            "Current state of service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )
        if c.json()["status"]["msg"] != "NotFound":
            self.clear()
            logging.warn(
                (
                    "Since service '{sname}' was still there, "
                    "it has been fully cleared: {json}"
                ).format(
                    sname=self.sname, json=json.dumps(c.json(), indent=2)
                )
            )

        logging.info(
            "Creating service '{sname}': {body}".format(
                sname=self.sname, body=json.dumps(body, indent=2)
            )
        )
        c = requests.put(
            "http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            ),
            json.dumps(body),
        )
        logging.info(
            "Reply from creating service '{sname}':\n  {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        train_data = [self.training_repo.value]
        parameters_input = {
            "test_split": self.tsplit.value,
            "shuffle": True,
            "db": True,
            "segmentation": True,
        }
        if self.testing_repo.value != "":
            train_data.append(self.testing_repo.value)
            parameters_input = {"shuffle": True}

        if self.multi_label.value:
            parameters_input["db"] = False
        parameters_mllib = {
            "gpu": True,
            "gpuid": self.gpuid.value,
            "resume": self.resume.value,
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
            "solver": {
                "test_initialization": self.test_init.value,
                "iterations": self.iterations.value,
                "test_interval": self.test_interval.value,
                "snapshot": self.snapshot_interval.value,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
                "iter_size": self.iter_size.value,
            },
        }
        ##TODO: lr policy as arguments
        # 'lr_policy':'step','stepsize':2000,'gamma':0.1,'snapshot':4000,'base_lr':args.base_lr,'solver_type':'SGD'}}
        if self.rand_skip.value > 0 and self.resume.value:
            parameters_mllib["solver"]["rand_skip"] = self.rand_skip.value
        if self.class_weights.value:
            parameters_mllib["class_weights"] = eval(self.class_weights.value)
        if self.ignore_label.value >= 0:
            parameters_mllib["ignore_label"] = self.ignore_label.value
        if self.timesteps.value:
            parameters_mllib["timesteps"] = self.timesteps.value

        parameters_output = {"measure": ["acc"]}

        body = {  # typing: Dict[str, Any]
            "service": self.sname,
            "async": True,
            "parameters": {
                "input": parameters_input,
                "mllib": parameters_mllib,
                "output": parameters_output,
            },
            "data": train_data,
        }

        logging.info(
            "Start training phase: {body}".format(
                body=json.dumps(body, indent=2)
            )
        )
        c = requests.post(
            "http://{host}:{port}/train".format(host=host, port=port),
            json.dumps(body),
        )
        logging.info(
            "Reply from training service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        print(json.dumps(c.json(), indent=2))



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
            layout=Layout(width="650px")
        )

        self._main_elt = HBox(
            [self._configuration, self._img_explorer],
            layout=Layout(width="900px"),
        )

        
        self.update_label_list(())

    @MLWidget.output.capture(clear_output=True)
    def run(self, *_) -> None:
        width = int(self.img_width.value)
        height = int(self.img_height.value)
        crop_size = int(self.crop_size.value)

        nclasses = int(self.nclasses.value)
        if nclasses == -1:
            nclasses = len(os.walk(self.training_repo.value).next()[1])

        host = self.host.value
        port = self.port.value
        description = "imagenet classifier"
        mllib = "caffe"

        model = {
            "templates": "../templates/caffe/",
            "repository": self.model_repo.value,
            "create_repository": True,
        }

        if self.weights.value:
            if not Path(self.model_repo.value).is_dir():
                logging.warn(
                    "Creating repository directory: {}".format(
                        self.model_repo.value
                    )
                )
                Path(self.model_repo.value).mkdir(parents=True)
                # change permission if dede is not run by current user
                Path(self.model_repo.value).chmod(0o777)

            shutil.copy(self.weights.value, self.model_repo.value + "/")

        parameters_input = {
            "connector": "image",
            "width": width,
            "height": height,
            "bw": self.bw.value,
            "db": True,
            "bbox": True
        }

        if self.multi_label.value:
            parameters_input["multi_label"] = True
            parameters_input["db"] = False
        if self.ctc.value:
            parameters_input["ctc"] = True
        if not self.finetune.value:
            if self.template.value:
                parameters_mllib = {
                    "template": self.template.value,
                    "nclasses": nclasses,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "layers": eval(self.layers.value),  # List of strings
                    "db": True,
                }
            else:
                parameters_mllib = {
                    "nclasses": nclasses,
                    "mirror": self.mirror.value,
                    "rotate": self.rotate.value,
                }
        else:
            if self.template.value:
                parameters_mllib = {
                    "template": self.template.value,
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": self.weights.value,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                }
            else:
                parameters_mllib = {
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": self.weights.value,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                }
        if self.multi_label.value:
            parameters_mllib["db"] = False

        if crop_size > 0:
            parameters_mllib["crop_size"] = crop_size
        if self.noise_prob.value > 0.0:
            parameters_mllib["noise"] = {
                "all_effects": True,
                "prob": self.noise_prob.value,
            }
        if self.distort_prob.value > 0.0:
            parameters_mllib["distort"] = {
                "all_effects": True,
                "prob": self.distort_prob.value,
            }
        parameters_mllib["gpu"] = True
        parameters_mllib["gpuid"] = self.gpuid.value
        if self.regression.value:
            parameters_mllib["regression"] = True

        parameters_output = {}
        # print (parameters_input)
        # print (parameters_mllib)
        # pserv = dd.put_service(self.sname.value,model,description,mllib,
        #                       parameters_input,parameters_mllib,parameters_output)

        body = { #: Dict[str, Any] = {
            "description": description,
            "mllib": mllib,
            "type": "supervised",
            "parameters": {
                "input": parameters_input,
                "mllib": parameters_mllib,
                "output": parameters_output,
            },
            "model": model,
        }

        c = requests.get(
            "http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            )
        )
        logging.info(
            "Current state of service '{sname}':\n  {json}".format(
                sname=self.sname, json=c.json()
            )
        )
        # useful for the clear() method
        if c.json()["status"]["msg"] != "NotFound":
            self.clear()
            logging.warn(
                (
                    "Since service '{sname}' was still there, "
                    "it has been fully cleared: {json}"
                ).format(sname=self.sname, json=c.json())
            )

        logging.warn(
            "Creating service '{sname}':  {body}".format(
                sname=self.sname, body=json.dumps(body, indent=2)
            )
        )
        c = requests.put(
            "http://{host}:{port}/services/{sname}".format(
                host=host, port=port, sname=self.sname
            ),
            json.dumps(body),
        )
        logging.warn(
            "Reply from creating service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        train_data = [self.training_repo.value]
        parameters_input = {
            "test_split": self.tsplit.value,
            "shuffle": True,
            "db": True,
        }
        if self.testing_repo.value != "":
            train_data.append(self.testing_repo.value)
            parameters_input = {"shuffle": True}

        if self.multi_label.value:
            parameters_input["db"] = False
        parameters_mllib = {
            "gpu": True,
            "gpuid": self.gpuid.value,
            "resume": self.resume.value,
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
            "solver": {
                "test_initialization": self.test_init.value,
                "iterations": self.iterations.value,
                "test_interval": self.test_interval.value,
                "snapshot": self.snapshot_interval.value,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
                "iter_size": self.iter_size.value,
            },
            "bbox": True
        }
        ##TODO: lr policy as arguments
        # 'lr_policy':'step','stepsize':2000,'gamma':0.1,'snapshot':4000,'base_lr':args.base_lr,'solver_type':'SGD'}}
        if self.rand_skip.value > 0 and self.resume.value:
            parameters_mllib["solver"]["rand_skip"] = self.rand_skip.value
        if self.class_weights.value:
            parameters_mllib["class_weights"] = eval(self.class_weights.value)
        if self.ignore_label.value >= 0:
            parameters_mllib["ignore_label"] = self.ignore_label.value
        if self.timesteps.value:
            parameters_mllib["timesteps"] = self.timesteps.value

        parameters_output = {"measure": ["map"]}


        body = {
            "service": self.sname,
            "async": True,
            "parameters": {
                "input": parameters_input,
                "mllib": parameters_mllib,
                "output": parameters_output,
            },
            "data": train_data,
        }

        logging.info("Start training phase: {body}".format(body=body))
        c = requests.post(
            "http://{host}:{port}/train".format(host=host, port=port),
            json.dumps(body),
        )
        logging.info(
            "Reply from training service '{sname}': {json}".format(
                sname=self.sname, json=json.dumps(c.json(), indent=2)
            )
        )

        print(json.dumps(c.json(), indent=2))