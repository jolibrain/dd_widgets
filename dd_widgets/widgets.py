# fmt: off
import logging

import json
from collections import OrderedDict
from datetime import timedelta
from inspect import signature
from pathlib import Path
from typing import Any, Dict, List, Optional, get_type_hints

from ipywidgets import (HTML, Button, Checkbox, FloatText, HBox, IntProgress,
                        IntText, Label, Layout, Output, SelectMultiple, Tab)
from ipywidgets import Text as TextWidget
from ipywidgets import VBox

from .core import JSONType, TalkWithDD
from .loghandler import OutputWidgetHandler
from .types import GPUIndex, GPUSelect, Solver, SolverDropdown, Engine, EngineDropdown

# fmt: on

info_loghandler = OutputWidgetHandler()


class RedirectOutput:
    """
    This class provide a decorator to redirect the output of functions to a
    specific output widget.
    """

    def __init__(
        self,
        output: Output,
        info_loghandler: Optional[OutputWidgetHandler] = None,
    ) -> None:
        self.output = output
        self.info_loghandler = info_loghandler

    def __call__(self, fun):
        def fun_wrapper(*args, **kwargs):
            if self.info_loghandler is not None:
                self.info_loghandler.out.clear_output()
            self.output.clear_output()
            with self.output:
                res = fun(*args, **kwargs)
                try:
                    print(json.dumps(res, indent=2))
                except Exception:
                    pass
                return res

        return fun_wrapper


# -- Core 'abstract' widget for many tasks


class BasicWidget:
    """This BasicWidget contains the most basicstest functionalities."""

    def __init__(self, *args, **kwargs):
        self.status_label = Label(
            value="Status: unknown", layout=Layout(margin="18px")
        )
        self.debug = HTML(
            layout={"width": "590px", "height": "800px", "border": "none"}
        )
        super().__init__(*args, **kwargs)

    def _ipython_display_(self):
        self._main_elt._ipython_display_()

    @property
    def status(self):
        return self.status_label.value

    @status.setter
    def status(self, value):
        label = []
        if "status" in value:
            label.append("status: {}".format(value["status"]))
        if "time" in value:
            label.append(
                "elapsed time: {}".format(timedelta(seconds=value["time"]))
            )

        self.status_label.value = ", ".join(label)

    def widgets_refresh(self, *_):
        from . import logfile_name

        with open(logfile_name, "r") as fh:
            l = fh.readlines()
            self.debug.value = (
                "<code style='display: block; white-space: pre-wrap;'>"
                + "".join(l[-200:])
                + "</code>"
            )


class JSONBuilder:
    def _create_model(self) -> JSONType:
        return {
            "templates": "../templates/caffe/",
            "repository": self.model_repo.value,
            "create_repository": True,
        }

    def _create_parameters_input(self) -> JSONType:
        raise NotImplementedError

    def _create_parameters_mllib(self) -> JSONType:
        return {}

    def _create_parameters_output(self) -> JSONType:
        return {"store_config": True}

    def _create_service_body(self) -> JSONType:
        return OrderedDict(
            [
                ("mllib", self.mllib.value),
                ("description", self.sname),
                ("type", self._type),
                (
                    "parameters",
                    {
                        "input": {
                            **self._create_parameters_input(),
                            **self._append_create_parameters_input,
                        },
                        "mllib": {
                            **self._create_parameters_mllib(),
                            **self._append_create_parameters_mllib,
                        },
                        "output": {
                            **self._create_parameters_output(),
                            **self._append_create_parameters_output,
                        },
                    },
                ),
                (
                    "model",
                    {**self._create_model(), **self._append_create_model},
                ),
            ]
        )

    def _train_parameters_input(self) -> JSONType:
        raise NotImplementedError

    def _train_parameters_mllib(self) -> JSONType:
        raise NotImplementedError

    def _train_parameters_output(self) -> JSONType:
        raise NotImplementedError

    def _train_data(self) -> List[str]:
        return [self.training_repo.value]

    def _train_service_body(self) -> JSONType:
        return OrderedDict(
            [
                ("service", self.sname),
                ("async", True),
                (
                    "parameters",
                    {
                        "input": {
                            **self._train_parameters_input(),
                            **self._append_train_parameters_input,
                        },
                        "mllib": {
                            **self._train_parameters_mllib(),
                            **self._append_train_parameters_mllib,
                        },
                        "output": {
                            **self._train_parameters_output(),
                            **self._append_train_parameters_output,
                        },
                    },
                ),
                ("data", self._train_data()),
            ]
        )


class MLWidget(TalkWithDD, JSONBuilder, BasicWidget):

    _type: str = "supervised"

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
        Engine: EngineDropdown,
        GPUIndex: GPUSelect,
    }

    def typing_info(self, local_vars: Dict[str, Any]):
        fun = self.__init__  # type: ignore
        typing_dict = get_type_hints(fun)
        for param in signature(fun).parameters.values():
            if param.name not in ["sname", "kwargs"]:
                yield (
                    param.name,
                    eval(param.name, local_vars),
                    typing_dict[param.name],
                )

    def on_start(self):
        self.value = self.iterations.value
        self.pbar.bar_style = "info"
        self.pbar.max = self.iterations.value

    def on_update(self, info):
        self.pbar.value = info["body"]["measure"].get("iteration", 0)
        self.pbar.bar_style = ""

    def on_finished(self, info):
        # a minima...
        self.pbar.value = self.iterations.value
        self.pbar.bar_style = "success"
        self.last_info = info

    def on_error(self, info):
        with self.output:
            logging.error(json.dumps(info, indent=2))
            raise RuntimeError(
                "Error code {code}: {msg}".format(
                    code=info["body"]["Error"]["dd_code"],
                    msg=info["body"]["Error"]["dd_msg"],
                )
            )

    def _train_data(self) -> List[str]:
        train_data = [self.training_repo.value]
        if self.testing_repo.value != "":
            train_data.append(self.testing_repo.value)
        return train_data

    def __init__(self, sname: str, local_vars: Dict[str, Any], *args) -> None:

        from . import logfile_name

        # logger.addHandler(log_viewer(self.output),)
        super().__init__(*args)
        kwargs = local_vars["kwargs"]

        self._append_create_parameters_input = kwargs.get(
            "create_parameters_input", {}
        )
        self._append_create_parameters_mllib = kwargs.get(
            "create_parameters_mllib", {}
        )
        self._append_create_parameters_output = kwargs.get(
            "create_parameters_output", {}
        )

        self._append_train_parameters_input = kwargs.get(
            "train_parameters_input", {}
        )
        self._append_train_parameters_mllib = kwargs.get(
            "train_parameters_mllib", {}
        )
        self._append_train_parameters_output = kwargs.get(
            "train_parameters_output", {}
        )

        self._append_create_model = kwargs.get("create_model", {})

        self.sname = sname
        self.output = Output(layout=Layout(max_width="650px"))
        self.pbar = IntProgress(
            min=0,
            max=100,
            description="Progression",
            layout=Layout(margin="18px"),
        )

        self.run_button = Button(description="Run training")
        self.resume_button = Button(description="Resume")
        self.info_button = Button(description="Info")
        self.delete_button = Button(description="Delete service")
        self.hardclear_button = Button(description="Hard clear")
        self.lightclear_button = Button(description="Light clear")

        self._widgets = [  # typing: List[Widget]
            HTML(
                value="<h2>{task} task: {sname}</h2>".format(
                    task=self.__class__.__name__, sname=self.sname
                )
            ),
            HBox([self.run_button, self.delete_button]),
            HBox([self.resume_button, self.hardclear_button]),
            HBox([self.info_button, self.lightclear_button]),
        ]

        self.run_button.on_click(RedirectOutput(self.output)(self.run))
        self.resume_button.on_click(RedirectOutput(self.output)(self.resume))
        self.info_button.on_click(RedirectOutput(self.output)(self.info))
        self.delete_button.on_click(
            RedirectOutput(self.output, info_loghandler)(self.delete)
        )
        self.hardclear_button.on_click(
            RedirectOutput(self.output, info_loghandler)(self.hardclear)
        )
        self.lightclear_button.on_click(
            RedirectOutput(self.output, info_loghandler)(self.lightclear)
        )

        for name, value, type_hint in self.typing_info(local_vars):
            self._add_widget(name, value, type_hint)

        self._configuration = VBox(
            self._widgets, layout=Layout(min_width="250px")
        )

        self._tabs = Tab(layout=Layout(height=""))
        self._output = VBox([HBox([self.pbar, self.status_label]), self._tabs])
        self._main_elt = HBox(
            [self._configuration, self._output], layout=Layout(width="1200px")
        )
        self._img_explorer = VBox(
            [self.output], layout=Layout(min_height="800px", width="590px")
        )

        self.refresh_button = Button(description="Refresh")
        self.refresh_button.on_click(
            RedirectOutput(self.output)(self.widgets_refresh)
        )
        self._tabs.children = [
            self._img_explorer,
            info_loghandler.out,
            VBox([self.refresh_button, self.debug]),
        ]
        self._tabs.set_title(0, "Exploration")
        self._tabs.set_title(1, "Logs (INFO)")
        self._tabs.set_title(2, f"{logfile_name.split('/')[-1]} (tail)")

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
            default_params = dict(
                value=type_hint() if value is None else (value),
                layout=Layout(width="100px", margin="4px 2px 4px 2px"),
            )
            if name == "gpuid":
                default_params["host"] = self.host.value
            setattr(self, name, widget_type(**default_params))

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

    def update_label_list(self, _):
        # should this be here?
        with self.output:
            if self.training_repo.value != "":
                self.train_labels.options = tuple(
                    sorted(
                        f.stem for f in Path(self.training_repo.value).glob("*")
                    )
                )
                self.train_labels.rows = min(10, len(self.train_labels.options))

            if self.testing_repo.value != "":
                self.test_labels.options = tuple(
                    sorted(
                        f.stem for f in Path(self.testing_repo.value).glob("*")
                    )
                )
                self.test_labels.rows = min(10, len(self.test_labels.options))
            if self.nclasses.value == -1:
                self.nclasses.value = str(len(self.train_labels.options))
