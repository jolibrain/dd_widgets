from pathlib import Path
from typing import List

import altair as alt
import pandas as pd
from ipywidgets import Button, HBox, SelectMultiple

from .mixins import ImageTrainerMixin
from .utils import sample_from_iterable
from .widgets import GPUIndex, MLWidget, Solver


class CSVTS(ImageTrainerMixin):
    def display_img(self, args):
        self.output.clear_output()
        with self.output:
            for csv in args["new"]:
                df = pd.read_csv(csv, sep=";")
                df.columns = df.columns.str.replace(".", "_")

                dropdown = alt.binding_select(options=list(df.columns[1:]))
                selection = alt.selection_single(
                    fields=["variable"],
                    bind=dropdown,
                    name="Selection of",  # empty=df.columns[1]
                )

                color = alt.condition(
                    selection, alt.Color("variable:N"), alt.value("lightgray")
                )
                scales = alt.selection_interval(encodings=["x"], bind="scales")
                chart = (
                    alt.Chart(df.melt("time"))
                    .mark_line()
                    .encode(x="time", y="value", color=color)
                    .add_selection(selection)
                    .transform_filter(selection)
                    .properties(width=400, height=300)
                    .add_selection(scales)
                )

                with alt.data_transformers.enable("default", max_rows=None):
                    display(chart)

    def update_train_file_list(self, *args):
        with self.output:
            self.file_list.options = [
                x.as_posix()
                for x in sample_from_iterable(
                    Path(self.training_repo.value).glob("*"), 10
                )
            ]

    def update_test_file_list(self, *args):
        with self.output:
            self.file_list.options = [
                x.as_posix()
                for x in sample_from_iterable(
                    Path(self.testing_repo.value).glob("*"), 10
                )
            ]

    def _create_parameters_input(self):
        return {
            "connector": "csvts",
            "db": False,
            "label": eval(self.label_columns.value),
            "ignore": eval(self.ignore_columns.value),
        }

    def _create_parameters_mllib(self):
        dic = dict(
            template="recurrent",
            regression=True,
            db=False,
            dropout=0.0,
            loss="L2",
        )
        dic["gpu"] = True
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic["gpuid"] = (
            list(self.gpuid.index)
            if len(self.gpuid.index) > 1
            else self.gpuid.index[0]
        )
        dic["layers"] = eval(self.layers.value)  #'["L50", "L50", "A3", "L3"]'
        return dic

    def _train_parameters_input(self):
        return {
            "shuffle": True,
            "separator": self.csv_separator.value,
            "db": False,
            "scale": True,
            "offset": 100,
        }

    def _train_parameters_mllib(self):

        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic = {
            "gpu": True,
            "gpuid": (
                list(self.gpuid.index)
                if len(self.gpuid.index) > 1
                else self.gpuid.index[0]
            ),
            "resume": self.resume.value,
            "timesteps": self.timesteps.value,
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
            "solver": {
                "iterations": self.iterations.value,
                "test_interval": self.test_interval.value,
                "snapshot": self.snapshot_interval.value,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
                "test_initialization": self.test_initialization.value,
            },
        }

        return dic

    def _train_parameters_output(self):
        return {"measure": ["L1"]}

    def __init__(
        self,
        sname: str,
        *,  # unnamed parameters are forbidden
        path: str = "",
        description: str = "Recurrent model",
        model_repo: Path = None,
        mllib: str = "caffe",
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        gpuid: GPUIndex = 0,
        nclasses: int = -1,
        label_columns: List[str] = [],
        ignore_columns: List[str] = [],
        layers: List[str] = [],
        csv_separator: str = ",",
        solver_type: Solver = "AMSGRAD",
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        resume: bool = False,
        base_lr: float = 1e-4,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        timesteps: int,
        test_initialization: bool = False,
        batch_size: int = 1000,
        test_batch_size: int = 100,
        **kwargs
    ):
        super().__init__(sname, locals())

        training_path = Path(self.training_repo.value)  # type: ignore

        if not training_path.exists():
            raise RuntimeError("Path {} does not exist".format(training_path))

        self.train_labels = Button(
            description=Path(self.training_repo.value).name  # type: ignore
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name  # type: ignore
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
