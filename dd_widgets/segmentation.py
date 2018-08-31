from pathlib import Path
from typing import List, Optional, Union

from IPython.display import display

from ipywidgets import Button, HBox

from .core import ImageTrainerMixin, img_handle, sample_from_iterable
from .widgets import MLWidget, Solver


class Segmentation(MLWidget, ImageTrainerMixin):
    def update_train_file_list(self, *args):
        with self.output:
            # print (Path(self.training_repo.value).read_text().split('\n'))
            self.file_dict = {
                Path(x.split()[0]): Path(x.split()[1])
                for x in Path(self.training_repo.value).read_text().split("\n")
                if len(x.split()) >= 2
            }

            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(self.file_dict.keys(), 10)
            ]

    def update_test_file_list(self, *args):
        with self.output:
            # print (Path(self.training_repo.value).read_text().split('\n'))
            self.file_dict = {
                Path(x.split()[0]): Path(x.split()[1])
                for x in Path(self.testing_repo.value).read_text().split("\n")
                if len(x.split()) >= 2
            }

            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(self.file_dict.keys(), 10)
            ]

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
        description: str = "Segmentation service",
        model_repo: Optional[str] = None,
        img_width: Optional[int] = None,
        img_height: Optional[int] = None,
        base_lr: float = 1e-3,
        iterations: int = 10000,
        snapshot_interval: int = 5000,
        test_interval: int = 1000,
        gpuid: Union[int, List[int]] = 0,
        layers: List[str] = [],
        template: Optional[str] = None,
        mirror: bool = True,
        rotate: bool = True,
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
        model_postfix: str = "",
        tboard: Optional[Path] = None,
        ignore_label: int = -1,
        multi_label: bool = False,
        regression: bool = False,
        rand_skip: int = 0,
        timesteps: int = 32,
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = "",
        loss: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = Button(
            description=Path(self.training_repo.value).name  # type: ignore
        )
        self.test_labels = Button(
            description=Path(self.testing_repo.value).name  # type: ignore
        )

        # self.testing_repo.observe(self.update_test_button, names="value")
        # self.training_repo.observe(self.update_train_button, names="value")

        self.train_labels.on_click(self.update_train_file_list)
        self.test_labels.on_click(self.update_test_file_list)

        self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        self.update_label_list(())

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
                _, img = img_handle(Path(path), self.file_dict[Path(path)])
                display(img)
                # display(Image(path))
                # integrate THIS : https://github.com/alx/react-bounding-box
                # (cv2.imread(self.file_dict[Path(path)].as_posix()))
