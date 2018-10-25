from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import pandas as pd
from ipywidgets import HTML

from .widgets import MLWidget


class TSNE_CSV(MLWidget):
    def __init__(
        self,
        sname: str,
        *,
        mllib: str = "tsne",
        training_repo: Path = None,
        description: str = "TSNE CSV service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        # gpuid: GPUIndex = 0,
        iterations: int = 5000,
        perplexity: int = 30,
        csv_id: str = "",
        csv_separator: str = ",",
        # csv_label: str = "",
    ) -> None:

        super().__init__(sname, locals())

        self.csv = pd.read_csv(training_repo)
        self.csv_label = self.csv.columns[0]
        self._displays = HTML(value=self.csv.sample(5)._repr_html_())
        self._img_explorer.children = [self._displays, self.output]

    def _create_service_body(self):
        body = OrderedDict(
            [
                ("mllib", self.mllib.value),
                ("description", self.sname),
                ("type", "unsupervised"),
                (
                    "parameters",
                    {
                        "input": {"connector": "csv"},
                        "mllib": {},
                        "output": {"store_config": True},
                    },
                ),
                (
                    "model",
                    {
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
                            "iterations": self.iterations.value,
                            "perplexity": self.perplexity.value,
                        },
                        "input": {
                            "label": self.csv_label,
                            "id": self.csv_id.value,
                            "separator": self.csv_separator.value,
                        },
                        "output": {},
                    },
                ),
                ("data", [self.training_repo.value]),
            ]
        )

        return body

    def plot(self, **kwargs):
        self.output.clear_output()
        with self.output:
            p = np.stack(
                x["vals"] for x in self.last_info["body"]["predictions"]
            )
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.scatter(*p.T, **kwargs)
            plt.close(fig)
            display(fig)

    def on_finished(self, info):
        with self.output:
            self.last_info = info
            self.plot()
