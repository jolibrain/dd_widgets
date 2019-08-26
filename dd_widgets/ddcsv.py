from pathlib import Path
from typing import List, Optional

import pandas as pd
from ipywidgets import HTML

from .core import JSONType
from .mixins import TextTrainerMixin
from .widgets import GPUIndex, Solver


class CSV(TextTrainerMixin):
    def __init__(
        self,
        sname: str,
        *,
        mllib: str = "caffe",
        training_repo: Path = None,
        testing_repo: Path = None,
        description: str = "CSV service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        gpuid: GPUIndex = 0,
        path: str = "",
        regression: bool = False,
        ntargets: int = 0,
        tsplit: float = 0.01,
        base_lr: float = 0.01,
        warmup_lr: float = 0.001,
        warmup_iter: int = 0,
        iterations: int = 100,
        test_interval: int = 1000,
        snapshot_interval: int = 1000,
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
        ignore_label: Optional[int] = -1,
        batch_size: int = 128,
        test_batch_size: int = 16,
        lregression: bool = False,
        scale: bool = False,
        csv_id: str = "",
        csv_separator: str = ",",
        csv_ignore: List[str] = [],
        csv_label: str = "",
        csv_label_offset: int = 0,
        csv_categoricals: List[str] = [],
        scale_pos_weight: float = 1.0,
        shuffle: bool = True,
        solver_type: Solver = "AMSGRAD",
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        rectified : bool = False,
        autoencoder: bool = False,
        class_weights: List[float] = [],
        target_repository: str = "",
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

        self._displays = HTML(
            value=pd.read_csv(training_repo).sample(5)._repr_html_()
        )
        self._img_explorer.children = [self._displays, self.output]

    def _create_parameters_input(self) -> JSONType:
        return {
            "connector": "csv",
            "labels": self.csv_label.value,
            "db": self.db.value,
        }

    def _train_parameters_input(self) -> JSONType:
        return {
            "autoencoder": self.autoencoder.value,
            "categoricals": eval(self.csv_categoricals.value),
            "db": self.db.value,
            "id": self.csv_id.value,
            "ignore": eval(self.csv_ignore.value),
            "label_offset": self.csv_label_offset.value,
            "label": self.csv_label.value,
            "scale": self.scale.value,
            "separator": self.csv_separator.value,
            "shuffle": self.shuffle.value,
            "test_split": self.tsplit.value,
        }
