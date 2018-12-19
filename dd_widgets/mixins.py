import logging
import os
import shutil
from pathlib import Path

from ipywidgets import Button, HBox, SelectMultiple

from .core import JSONType
from .widgets import MLWidget
from .utils import sample_from_iterable


class TextTrainerMixin(MLWidget):
    def __init__(self, *args) -> None:
        super().__init__(*args)

    def _create_parameters_mllib(self) -> JSONType:

        dic = {
            "nclasses": self.nclasses.value,
            "activation": self.activation.value,
            "db": self.db.value,
            "dropout": self.dropout.value,
            "template": self.template.value,
            "layers": eval(self.layers.value),
            "autoencoder": self.autoencoder.value,
            "regression": self.regression.value,
        }

        if self.template.value is not None:
            del dic["template"]

        if self.regression.value:
            del dic["nclasses"]
            dic["ntargets"] = int(self.ntargets.value)

        if self.mllib.value == "xgboost":
            del dic["solver"]
            dic["iterations"] = self.iterations.value
            dic["db"] = False

        if self.lregression.value:
            dic["template"] = "lregression"
            del dic["layers"]

        if self.finetune.value:
            dic["finetuning"] = True
            dic["weights"] = self.weights.value

        return dic

    def _train_parameters_mllib(self) -> JSONType:
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic = {
            "gpu": True,
            "gpuid": (
                list(self.gpuid.index)
                if len(self.gpuid.index) > 1
                else self.gpuid.index[0]
            ),
            "resume": self.resume.value,
            "solver": {
                "iterations": self.iterations.value,
                "iter_size": 1,
                "snapshot_interval": self.snapshot_interval.value,
                "test_interval": self.test_interval.value,
                "test_initialization": False,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
            },
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
        }

        if self.ignore_label.value != -1:
            dic["ignore_label"] = int(self.ignore_label.value)

        if self.mllib.value == "xgboost":
            del dic["solver"]
            dic["iterations"] = self.iterations.value
            dic["objective"] = self.objective.value
            dic["booster_params"] = {
                "scale_pos_weight": self.scale_pos_weight.value
            }

        if self.class_weights.value:
            dic['class_weights'] = eval(self.class_weights.value)

        return dic

    def _train_parameters_output(self) -> JSONType:
        dic = {"measure": ["cmdiag", "cmfull", "mcll", "f1"]}

        if self.regression.value:
            del dic["measure"]
            dic["measure"] = ["eucll"]

        if self.nclasses.value == 2:
            dic["measure"].append("auc")

        if self.autoencoder.value:
            dic["measure"] = ["eucll"]

        return dic


class ImageTrainerMixin(MLWidget):
    def __init__(self, *args) -> None:
        super().__init__(*args)

        training_path = Path(self.training_repo.value)  # type: ignore

        if not training_path.exists():
            raise RuntimeError("Path {} does not exist".format(training_path))

        if training_path.is_dir():

            self.train_labels = SelectMultiple(
                options=[],
                value=[],
                description="Training labels",
                disabled=False,
            )

            self.test_labels = SelectMultiple(
                options=[],
                value=[],
                description="Testing labels",
                disabled=False,
            )

            self.testing_repo.observe(  # type: ignore
                self.update_label_list, names="value"
            )
            self.training_repo.observe(  # type: ignore
                self.update_label_list, names="value"
            )

            self.train_labels.observe(self.update_train_dir_list, names="value")
            self.test_labels.observe(self.update_test_dir_list, names="value")
            self.file_list.observe(self.display_img, names="value")

        else:
            self.train_labels = Button(
                description=Path(self.training_repo.value).name  # type: ignore
            )
            self.test_labels = Button(
                description=Path(self.testing_repo.value).name  # type: ignore
            )

            self.train_labels.on_click(self.update_train_file_list)
            self.test_labels.on_click(self.update_test_file_list)
            self.file_list.observe(self.display_img, names="value")

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        self.update_label_list(())

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

    def update_train_dir_list(self, *args):
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

    def update_test_dir_list(self, *args):
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

    def _create_parameters_input(self) -> JSONType:

        width = int(self.img_width.value)
        height = int(self.img_height.value)

        if self.weights.value:
            if not Path(self.model_repo.value).is_dir():
                logging.warn(
                    "Creating repository directory: {}".format(
                        self.model_repo.value
                    )
                )
                Path(self.model_repo.value).mkdir(parents=True)
                # change permission if dede is not run by current user
                Path(self.model_repo.value).chmod(0o777)

            shutil.copy(self.weights.value, self.model_repo.value + "/")

        parameters_input = {
            "connector": "image",
            "width": width,
            "height": height,
            "bw": self.bw.value,
            "db": True,
        }

        if self.unchanged_data:
            parameters_input["unchanged_data"] = True

        if self.__class__.__name__ == "Detection":
            parameters_input["bbox"] = True

        if self.__class__.__name__ == "Segmentation":
            parameters_input["segmentation"] = True

        if self.multi_label.value:
            parameters_input["multi_label"] = True
            parameters_input["db"] = False

        if self.ctc.value:
            parameters_input["ctc"] = True

        return parameters_input

    def _create_parameters_mllib(self) -> JSONType:

        nclasses = int(self.nclasses.value)
        if nclasses == -1:
            nclasses = len(os.walk(self.training_repo.value).next()[1])

        logging.info("{} classes".format(nclasses))

        if not self.finetune.value:
            if self.template.value:
                dic = {
                    "template": self.template.value,
                    "nclasses": nclasses,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "scale": self.scale.value,
                    "layers": eval(self.layers.value),  # List of strings
                    "db": True,
                    "activation": self.activation.value,
                    "dropout": self.dropout.value,
                    "autoencoder": self.autoencoder.value,
                }
            else:
                dic = {
                    "nclasses": nclasses,
                    "mirror": self.mirror.value,
                    "rotate": self.rotate.value,
                    "scale": self.scale.value,
                    "autoencoder": self.autoencoder.value,
                }
        else:
            if self.template.value:
                dic = {
                    "template": self.template.value,
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": self.weights.value,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "scale": self.scale.value,
                    "layers": eval(self.layers.value),  # List of strings
                    "db": True,
                    "activation": self.activation.value,
                    "dropout": self.dropout.value,
                    "autoencoder": self.autoencoder.value,
                }
            else:
                dic = {
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": self.weights.value,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "scale": self.scale.value,
                    "autoencoder": self.autoencoder.value,
                }
        if self.multi_label.value:
            dic["db"] = False

        crop_size = int(self.crop_size.value)
        if crop_size > 0:
            dic["crop_size"] = crop_size
        if self.noise_prob.value > 0.0:
            dic["noise"] = {"all_effects": True, "prob": self.noise_prob.value}
        if self.distort_prob.value > 0.0:
            dic["distort"] = {
                "all_effects": True,
                "prob": self.distort_prob.value,
            }
        # if any(
        #     [
        #         self.all_effects.value,
        #         self.persp_horizontal.value,
        #         self.persp_vertical.value,
        #         self.zoom_out.value,
        #         self.zoom_in.value,
        #     ]
        # ) or any(
        #     p != ""
        #     for p in [
        #         self.persp_factor.value,
        #         self.zoom_factor.value,
        #         self.pad_mode.value,
        #         self.prob.value,
        #     ]
        # ):
        #     dic["geometry"] = {}
        #     # -- booleans --
        #     if self.all_effects.value:
        #         dic["geometry"]["all_effects"] = True
        #     if self.persp_horizontal.value:
        #         dic["geometry"]["persp_horizontal"] = True
        #     if self.persp_vertical.value:
        #         dic["geometry"]["persp_vertical"] = True
        #     if self.zoom_out.value:
        #         dic["geometry"]["zoom_out"] = True
        #     if self.zoom_in.value:
        #         dic["geometry"]["zoom_in"] = True
        #     # -- strings --
        #     if self.pad_mode.value != "":
        #         dic["geometry"]["pad_mode"] = float(self.pad_mode.value)
        #     # -- float --
        #     if self.persp_factor.value != "":
        #         dic["geometry"]["persp_factor"] = float(self.persp_factor.value)
        #     if self.zoom_factor.value != "":
        #         dic["geometry"]["zoom_factor"] = float(self.zoom_factor.value)
        #     if self.prob.value != "":
        #         dic["geometry"]["prob"] = float(self.prob.value)

        dic["gpu"] = True
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic["gpuid"] = (
            list(self.gpuid.index)
            if len(self.gpuid.index) > 1
            else self.gpuid.index[0]
        )
        if self.regression.value:
            dic["regression"] = True

        return dic

    def _train_parameters_input(self) -> JSONType:

        dic = {"test_split": self.tsplit.value, "shuffle": True, "db": True}

        if self.multi_label.value:
            dic["db"] = False

        if self.ctc.value:
            if self.align.value:
                dic["align"] = True

        return dic

    def _train_parameters_mllib(self) -> JSONType:
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        dic = {
            "gpu": True,
            "gpuid": (
                list(self.gpuid.index)
                if len(self.gpuid.index) > 1
                else self.gpuid.index[0]
            ),
            "resume": self.resume.value,
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
            "solver": {
                "test_initialization": self.test_init.value,
                "iterations": self.iterations.value,
                "test_interval": self.test_interval.value,
                "snapshot": self.snapshot_interval.value,
                "base_lr": self.base_lr.value,
                "solver_type": self.solver_type.value,
                "iter_size": self.iter_size.value,
            },
        }
        if self.__class__.__name__ == "Detection":
            dic["bbox"] = True

        if self.rand_skip.value > 0 and self.resume.value:
            dic["solver"]["rand_skip"] = self.rand_skip.value
        if self.class_weights.value:
            dic["class_weights"] = eval(self.class_weights.value)
        if self.ignore_label.value >= 0:
            dic["ignore_label"] = int(self.ignore_label.value)
        if self.timesteps.value:
            dic["timesteps"] = self.timesteps.value

        return dic

    def _train_parameters_output(self) -> JSONType:

        if self.multi_label.value and self.regression.value:
            dic = {
                "measure": [
                    "kl",
                    "js",
                    "was",
                    "ks",
                    "dc",
                    "r2",
                    "deltas",
                    "eucll",
                ]
            }
        else:
            dic = {"measure": ["mcll", "f1", "acc-5"]}

        # special cases
        if self.ctc.value:
            dic = {"measure": ["acc"]}
        elif self.autoencoder.value:
            dic = {"measure": ["eucll"]}

        dic["target_repository"] = self.target_repository.value

        return dic
