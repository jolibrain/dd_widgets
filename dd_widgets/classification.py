from pathlib import Path
from typing import List, Optional, Union

from IPython.display import display

from ipywidgets import HBox, SelectMultiple

from .core import ImageTrainerMixin, img_handle, sample_from_iterable
from .widgets import MLWidget, Solver


class Classification(MLWidget, ImageTrainerMixin):
    ctc = False

    def update_train_file_list(self, *args):
        with self.output:
            if len(self.train_labels.value) == 0:
                return
            directory = (
                Path(self.training_repo.value) / self.train_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.test_labels.value = []

    def update_test_file_list(self, *args):
        with self.output:
            if len(self.test_labels.value) == 0:
                return
            directory = (
                Path(self.testing_repo.value) / self.test_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.train_labels.value = []

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
        training_repo: Path = None,
        testing_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        nclasses: int = -1,
        description: str = "classification service",
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-4,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        gpuid: Union[int, List[int]] = 0,
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
        regression: bool = False,
        rand_skip: int = 0,
        timesteps: int = 32,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = SelectMultiple(
            options=[], value=[], description="Training labels", disabled=False
        )

        self.test_labels = SelectMultiple(
            options=[], value=[], description="Testing labels", disabled=False
        )

        self.testing_repo.observe(  # type: ignore
            self.update_label_list, names="value"
        )
        self.training_repo.observe(  # type: ignore
            self.update_label_list, names="value"
        )

        self.train_labels.observe(self.update_train_file_list, names="value")
        self.test_labels.observe(self.update_test_file_list, names="value")
        self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        self.update_label_list(())
