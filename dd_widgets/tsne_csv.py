from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

import pandas as pd
from ipywidgets import HTML

from .core import JSONType
from .widgets import GPUIndex, MLWidget


class TSNE_CSV(MLWidget):
    _type = "unsupervised"

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
        gpuid: GPUIndex = 0,
        # -- tsne specific
        iterations: int = 5000,
        perplexity: int = 30,
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

        self.csv = pd.read_csv(training_repo)
        self.csv_label = self.csv.columns[0]
        self._displays = HTML(value=self.csv.sample(5)._repr_html_())
        self._img_explorer.children = [self._displays, self.output]

    def _create_parameters_input(self) -> JSONType:
        return {"connector": "csv"}

    def _train_parameters_mllib(self) -> JSONType:
        return {
            "iterations": self.iterations.value,
            "perplexity": self.perplexity.value,
        }

    def _train_parameters_input(self) -> JSONType:
        return {
            "label": self.csv_label,
            "id": self.csv_id.value,
            "separator": self.csv_separator.value,
        }

    def plot(self, **kwargs):
        p = np.stack(x["vals"] for x in self.last_info["body"]["predictions"])
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(*p.T, **kwargs)
        plt.close(fig)
        display(fig)

    def on_finished(self, info):
        self.output.clear_output()
        with self.output:
            self.last_info = info
            self.plot()
