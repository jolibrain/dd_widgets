from pathlib import Path
from typing import List, Optional

from IPython.display import display

import cv2

from .core import JSONType
from .mixins import ImageTrainerMixin
from .utils import img_handle
from .widgets import GPUIndex, Solver, Engine


class Classification(ImageTrainerMixin):

    def display_img(self, args):
        imread_args = tuple()
        if self.unchanged_data.value:
            imread_args = (cv2.IMREAD_UNCHANGED,)
        self.output.clear_output()
        with self.output:
            for path in args["new"]:
                shape, img = img_handle(Path(path), imread_args=imread_args)
                if self.img_width.value == "":
                    self.img_width.value = str(shape[0])
                if self.img_height.value == "":
                    self.img_height.value = str(shape[1])
                display(
                    img
                )  # TODO display next to each other with shape info as well

    def __init__(  # type: ignore
        self,
        sname: str,
        *,  # unnamed parameters are forbidden
        mllib: str = "caffe",
        engine: Engine = "DEFAULT",
        training_repo: Path = None,
        testing_repo: Path = None,
        description: str = "classification service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        gpuid: GPUIndex = 0,
        # -- specific
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
        # -- geometry --
        #all_effects: bool = False,
        persp_horizontal: bool = False,
        persp_vertical: bool = False,
        zoom_out: bool = False,
        zoom_in: bool = False,
        pad_mode: str = "",
        persp_factor: str = "",
        zoom_factor: str = "",
        geometry_prob: str = "",
        # -- / geometry --
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = "",
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

    def _train_parameters_output(self) -> JSONType:
        dic = super()._train_parameters_output()
        dic["measure"] = ["mcll", "f1", "acc-5", "cmdiag"]
        return dic
