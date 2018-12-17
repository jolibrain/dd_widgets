
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ipywidgets import HTML

from .widgets import MLWidget


class TSNE_Text(MLWidget):
    def __init__(
        self,
        sname: str,
        *,
        mllib: str = "tsne",
        training_repo: Path = None,
        description: str = "TSNE Text service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        # gpuid: GPUIndex = 0,
        iterations: int = 5000,
        perplexity: int = 30,
    ) -> None:

        super().__init__(sname, locals())

        self._displays = HTML()
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
                        "input": {"connector": "txt"},
                        "mllib": {},
                        "output": {},
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

    def _train_service_body(self):

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
                            "min_count": 10,
                            "min_word_length": 5,
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
