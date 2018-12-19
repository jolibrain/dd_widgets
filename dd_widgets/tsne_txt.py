from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display

from ipywidgets import HTML

from .core import JSONType
from .widgets import GPUIndex, MLWidget


class TSNE_Text(MLWidget):
    _type = "unsupervised"

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
        gpuid: GPUIndex = 0,
        # -- tsne specific
        iterations: int = 5000,
        perplexity: int = 30,
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

        self._displays = HTML()
        self._img_explorer.children = [self._displays, self.output]

    def _create_parameters_input(self) -> JSONType:
        return {"connector": "txt"}

    def _train_parameters_input(self) -> JSONType:
        return {"min_count": 10, "min_word_length": 5}

    def _train_parameters_mllib(self) -> JSONType:
        return {
            "iterations": self.iterations.value,
            "perplexity": self.perplexity.value,
        }

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
