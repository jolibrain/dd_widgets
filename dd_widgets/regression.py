from pathlib import Path
from typing import List, Optional

from IPython.display import display

import cv2

from .core import JSONType
from .mixins import ImageTrainerMixin
from .utils import img_handle
from .widgets import GPUIndex, Solver, Engine


class Regression(ImageTrainerMixin):
    def display_img(self, args):
        self.output.clear_output()
        imread_args = tuple()
        if self.unchanged_data.value:
            imread_args = (cv2.IMREAD_UNCHANGED,)
        with self.output:
            for path in args["new"]:
                shape, img = img_handle(path=Path(path), imread_args=imread_args)
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
        engine: Engine = "CUDNN_SINGLE_HANDLE",
        training_repo: Path = None,
        testing_repo: Path = None,
        description: str = "classification service",
        model_repo: Optional[str] = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        gpuid: GPUIndex = 0,
        # -- specific
        ntargets: int = 1,
        nclasses: int = -1,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-4,
        lr_policy: str = "fixed",
        stepvalue: List[int] = [],
        warmup_lr: float = 1e-5,
        warmup_iter: int = 0,
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
        persp_horizontal: bool = None,
        persp_vertical: bool = None,
        zoom_out: bool = None,
        zoom_in: bool = None,
        pad_mode: str = None,
        persp_factor: str = None,
        zoom_factor: str = None,
        geometry_prob: str = None,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: Solver = "SGD",
        sam : bool = False,
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        rectified : bool = False,
        decoupled_wd_periods : int = 4,
        decoupled_wd_mult : float = 2.0,
        lr_dropout : float = 1.0,
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
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = "",
        loss: str = "L2",
        **kwargs
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
        dic["finetuning"] = self.finetune.value
        dic["loss"] = self.loss.value
        return dic

    def _train_parameters_input(self) -> JSONType:
        dic = super()._train_parameters_input()
        dic["db"] = False
        return dic

    def _train_parameters_mllib(self) -> JSONType:
        dic = super()._train_parameters_mllib()
        dic["db"] = False
        return dic

    def _train_parameters_output(self) -> JSONType:
        dic = super()._train_parameters_output()
        dic["measure"] = ["eucll"]
        return dic

    def _create_parameters_output(self) -> JSONType:
        dic = super()._create_parameters_output()
        dic["measure"] = ["eucll"]
        return dic
