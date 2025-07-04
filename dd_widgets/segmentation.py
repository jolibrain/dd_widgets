from pathlib import Path
from typing import List, Optional

from IPython.display import display

import cv2

from .core import JSONType
from .mixins import ImageTrainerMixin
from .utils import img_handle
from .widgets import GPUIndex, Solver, Engine


class Segmentation(ImageTrainerMixin):
    def __init__(
        self,
        sname: str,
        *,  # unnamed parameters are forbidden
        mllib: str = "caffe",
        engine: Engine = "CUDNN_SINGLE_HANDLE",
        training_repo: Path = None,
        testing_repo: List[Path] = None,
        description: str = "Segmentation service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        gpuid: GPUIndex = 0,
        # -- specific
        nclasses: int = -1,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-3,
        lr_policy: str = "fixed",
        stepvalue: List[int] = [],
        warmup_lr: float = 0.0001,
        warmup_iter: int = 0,
        db: bool = False,
        iterations: int = 10000,
        activation: str = "relu",
        dropout: float = 0.0,
        autoencoder: bool = False,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        layers: List[str] = [],
        template: Optional[str] = None,
        mirror: bool = True,
        rotate: bool = True,
        mean: List[float] = [],
        std: List[float] = [],
        scale: float = 1.0,
        rgb: bool = False,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        histogram_equalization: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: Solver = "SGD",
        sam : bool = False,
        swa : bool = False,
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        rectified : bool = False,
        decoupled_wd_periods : int = 4,
        decoupled_wd_mult : float = 2.0,
        lr_dropout : float = 1.0,
        noise_prob: float = 0.0,
        distort_prob: float = 0.0,
        cutout_prob: float = 0.0,
        # -- geometry --
        # all_effects: bool = False,
        persp_horizontal: bool = False,
        persp_vertical: bool = False,
        transl_horizontal: bool = False,
        transl_vertical: bool = False,
        zoom_out: bool = False,
        zoom_in: bool = False,
        pad_mode: str = "CONSTANT",
        persp_factor: float = 0.25,
        transl_factor: float = 0.5,
        zoom_factor: float = 0.25,
        geometry_prob: float = 0.0,
        # -- / geometry --
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        model_postfix: str = "",
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = "",
        loss: str = "",
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

    def display_img(self, args):
        self.output.clear_output()
        imread_args = tuple()
        if self.unchanged_data.value:
            imread_args = (cv2.IMREAD_UNCHANGED,)
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

                _, img = img_handle(
                    Path(path),
                    self.file_dict[Path(path)],
                    nclasses=self.nclasses.value,
                )
                display(img)
                # display(Image(path))
                # integrate THIS : https://github.com/alx/react-bounding-box
                # (cv2.imread(self.file_dict[Path(path)].as_posix()))

    def _create_parameters_input(self) -> JSONType:
        dic = super()._create_parameters_input()
        dic["segmentation"] = True
        dic["db"] = self.db.value
        return dic

    def _create_parameters_mllib(self) -> JSONType:
        dic = super()._create_parameters_mllib()
        dic["loss"] = self.loss.value
        dic["segmentation"] = True
        return dic

    def _train_parameters_input(self) -> JSONType:
        dic = super()._train_parameters_input()
        dic["segmentation"] = True
        dic["std"] = eval(self.std.value)
        dic["scale"] = self.scale.value
        dic["mean"] = eval(self.mean.value)
        return dic

    def _train_parameters_mllib(self) -> JSONType:
        dic = super()._train_parameters_mllib()
        dic["segmentation"] = True
        return dic
    
    def _train_parameters_output(self) -> JSONType:
        dic = super()._train_parameters_output()
        dic["measure"] = ["acc"]
        return dic
