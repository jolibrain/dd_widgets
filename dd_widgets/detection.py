from pathlib import Path
from typing import List, Optional

from IPython.display import display

import cv2

from .core import JSONType
from .mixins import ImageTrainerMixin
from .utils import img_handle
from .widgets import GPUIndex, Solver, Engine


class Detection(ImageTrainerMixin):

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
                _, img = img_handle(
                    Path(path),
                    bbox=self.file_dict[Path(path)],
                    nclasses=self.nclasses.value
                )
                display(img)

    def __init__(
        self,
        sname: str,
        *,  # unnamed parameters are forbidden
        mllib: str = "caffe",
        engine: Engine = "DEFAULT",
        training_repo: Path = None,
        testing_repo: Path = None,
        description: str = "Detection service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        gpuid: GPUIndex = 0,
        # -- specific
        nclasses: int = -1,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        db_width: int = 0,
        db_height: int = 0,
        base_lr: float = 1e-4,
        warmup_lr: float = 1e-5,
        warmup_iter: int = 0,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        layers: List[str] = [],
        activation: Optional[str] = "relu",
        dropout: float = 0.0,
        autoencoder: bool = False,
        template: Optional[str] = None,
        mirror: bool = False,
        rotate: bool = False,
        scale: float = 1.0,
        all_effects: bool = False,
        persp_horizontal: bool = True,
        persp_vertical: bool = True,
        zoom_out: bool = False,
        zoom_in: bool = False,
        pad_mode: str = "MIRRORED",
        persp_factor: float = 0.25,
        zoom_factor: float = 0.25,
        geometry_prob: float = 0.0,
        tsplit: float = 0.0,
        finetune: bool = False,
        resume: bool = False,
        bw: bool = False,
        crop_size: int = -1,
        batch_size: int = 32,
        test_batch_size: int = 16,
        iter_size: int = 1,
        solver_type: Solver = "SGD",
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        rectified : bool = False,
        decoupled_wd_periods : int = 4,
        decoupled_wd_mult : float = 2.0,
        lr_dropout : float = 1.0,
        noise_prob: float = 0.001,
        distort_prob: float = 0.5,
        test_init: bool = False,
        class_weights: List[float] = [],
        weights: Path = None,
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        unchanged_data: bool = False,
        target_repository: str = "",
        ctc: bool = False,
        ssd_expand_prob: float = -1.0,
        ssd_max_expand_ratio: float = -1.0,
        ssd_mining_type: str = "",
        ssd_neg_pos_ratio: float = -1.0,
        ssd_neg_overlap: float = -1.0,
        ssd_keep_top_k: int = -1,
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

    def _create_parameters_input(self) -> JSONType:
        dic = super()._create_parameters_input()
        dic['bbox'] = True
        return dic

    def _create_parameters_mllib(self) -> JSONType:
        dic = super()._create_parameters_mllib()
        net = {'ssd_expand_prob':self.ssd_expand_prob.value,
               'ssd_max_expand_ratio':self.ssd_max_expand_ratio.value,
               'ssd_mining_type':self.ssd_mining_type.value,
               'ssd_neg_pos_ratio':self.ssd_neg_pos_ratio.value,
               'ssd_neg_overlap':self.ssd_neg_overlap.value,
               'ssd_keep_top_k':self.ssd_keep_top_k.value}
        dic.update(net)
        return dic
    
    def _train_parameters_input(self) -> JSONType:
        dic = super()._train_parameters_input()
        dic["db_width"] = self.db_width.value
        dic["db_height"] = self.db_height.value
        return dic

    def _train_parameters_mllib(self) -> JSONType:
        dic = super()._train_parameters_mllib()
        dic['bbox'] = True
        return dic

    def _train_parameters_output(self) -> JSONType:
        dic = super()._train_parameters_output()
        dic['measure'] = ['map']
        return dic
