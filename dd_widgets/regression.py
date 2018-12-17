from pathlib import Path
from typing import List, Optional

from IPython.display import display

from .core import JSONType
from .mixins import ImageTrainerMixin
from .utils import img_handle
from .widgets import GPUIndex, Solver


class Regression(ImageTrainerMixin):
    def display_img(self, args):
        self.output.clear_output()
        with self.output:
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
        *,  # unnamed parameters are forbidden
        mllib: str = "caffe",
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        gpuid: GPUIndex = 0,
        path: str = "",
        ntargets: int = 1,
        nclasses: int = -1,
        description: str = "classification service",
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-4,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        layers: List[str] = [],
        template: Optional[str] = None,
        activation: Optional[str] = "relu",
        dropout: float = 0.0,
        autoencoder: bool = False,
        mirror: bool = False,
        rotate: bool = False,
        scale: float = 1.0,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: Solver = "SGD",
        noise_prob: float = 0.0,
        distort_prob: float = 0.0,
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = True,
        rand_skip: int = 0,
        timesteps: int = 32,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

    def _create_parameters_input(self) -> JSONType:
        dic = super()._create_parameters_input()
        dic["db"] = False
        return dic

    def _create_parameters_mllib(self) -> JSONType:
        dic = super()._create_parameters_mllib()
        del dic["nclasses"]
        dic["db"] = False
        dic["ntargets"] = int(self.ntargets.value)
        dic["finetuning"] = False
        return dic

    def _train_parameters_output(self) -> JSONType:
        dic = super()._train_parameters_output()
        dic["measure"] = ["eucll"]
        return dic
