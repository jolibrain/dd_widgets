from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import pandas as pd
from ipywidgets import HTML

from .widgets import MLWidget, Solver


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
