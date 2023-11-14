import logging
import os
import shutil
import requests
from pathlib import Path

from ipywidgets import Button, HBox, SelectMultiple

from .core import JSONType
from .widgets import MLWidget
from .utils import sample_from_iterable, is_url


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
            "engine": self.engine.value
        }

        if self.template.value is None:
            del dic["template"]

        if self.regression.value:
            del dic["nclasses"]
            dic["ntargets"] = int(self.ntargets.value)

        if self.mllib.value == "xgboost":
            if "solver" in dic:
                del dic["solver"]
            dic["iterations"] = self.iterations.value
            dic["db"] = False
        elif self.mllib.value == "torch":
            dic["gpu"] = True # force true at service creation with torch

        if self.lregression.value:
            dic["template"] = "lregression"
            del dic["layers"]

        if self.finetune.value:
            dic["finetuning"] = True
            if self.weights.value:
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
                "iter_size": self.iter_size.value,
                "snapshot_interval": self.snapshot_interval.value,
                "test_interval": self.test_interval.value,
                "test_initialization": False,
                "base_lr": self.base_lr.value,
                "warmup_lr": self.warmup_lr.value,
                "warmup_iter": self.warmup_iter.value,
                "weight_decay": self.weight_decay.value,
                "solver_type": self.solver_type.value,
                "sam": self.sam.value,
                "swa": self.swa.value,
                "lookahead": self.lookahead.value,
                "lookahead_steps": self.lookahead_steps.value,
                "lookahead_alpha": self.lookahead_alpha.value,
                "rectified": self.rectified.value,
                "decoupled_wd_periods": self.decoupled_wd_periods.value,
                "decoupled_wd_mult": self.decoupled_wd_mult.value,
                "lr_dropout": self.lr_dropout.value,
                "lr_policy": self.lr_policy.value,
            },
            "net": {
                "batch_size": self.batch_size.value,
                "test_batch_size": self.test_batch_size.value,
            },
        }

        if self.stepvalue.value:
            dic["solver"]["stepvalue"] = eval(self.stepvalue.value)
        
        if self.ignore_label.value != -1:
            dic["ignore_label"] = int(self.ignore_label.value)

        if self.mllib.value == "xgboost":
            del dic["solver"]
            dic["iterations"] = self.iterations.value
            dic["objective"] = "multi:softprob" # default
            # unset yet
            #dic["booster_params"] = {
            #    "scale_pos_weight": self.scale_pos_weight.value
            #}

        if self.class_weights.value:
            dic["class_weights"] = eval(self.class_weights.value)

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

            if not "," in self.testing_repo.value:
                self.testing_repo.value = "['" + self.testing_repo.value + "']"

            self.testing_repo.observe(  # type: ignore
                self.update_label_list, names="value"
            )
            self.training_repo.observe(  # type: ignore
                self.update_label_list, names="value"
            )

            self.train_labels.observe(self.update_train_dir_list, names="value")
            self.test_labels.observe(self.update_test_dir_list, names="value")
            self.file_list.observe(self.display_img, names="value")

            labels_hbox = HBox([self.train_labels, self.test_labels])
        else:
            self.train_labels = Button(
                description=Path(self.training_repo.value).name  # type: ignore
            )
            if not "," in self.testing_repo.value:
                self.testing_repo.value = "['" + self.testing_repo.value + "']"

            test_files = eval(self.testing_repo.value)
            self.test_labels = []

            for i in range(len(test_files)):
                test_file = test_files[i]
                self.test_labels.append(
                    Button(description=Path(test_file).name)  # type: ignore
                )
                self.test_labels[-1].on_click(self.get_update_test_file_list(i))

            self.train_labels.on_click(self.update_train_file_list)
            self.file_list.observe(self.display_img, names="value")

            labels_hbox = HBox([self.train_labels, *self.test_labels])

        self._img_explorer.children = [
            HBox([labels_hbox]),
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

    def update_test_file_list(self, test_id, *args):
        with self.output:
            # print (Path(self.training_repo.value).read_text().split('\n'))
            self.file_dict = {
                Path(x.split()[0]): Path(x.split()[1])
                for x in Path(eval(self.testing_repo.value)[test_id]).read_text().split("\n")
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
                Path(eval(self.testing_repo.value)[0]) / self.test_labels.value[0]
            )
            self.file_list.options = [
                fh.as_posix()
                for fh in sample_from_iterable(directory.glob("**/*"), 10)
            ]
            self.train_labels.value = []

    def _init_repository(self, resume: bool = None):
        if self.weights.value and not self.resume.value:
            if not Path(self.model_repo.value).is_dir():
                logging.warn(
                    "Creating repository directory: {}".format(
                        self.model_repo.value
                    )
                )
                Path(self.model_repo.value).mkdir(parents=True)
                # change permission if dede is not run by current user
                Path(self.model_repo.value).chmod(0o777)

            if is_url(self.weights.value):
                filename = self.weights.value.split("/")[-1]
                r = requests.get(self.weights.value, allow_redirects=True)

                with open(os.path.join(self.model_repo.value, filename), "wb") as weights_file:
                    weights_file.write(r.content)
            else:
                shutil.copy(self.weights.value, self.model_repo.value + "/")

    def _create_parameters_input(self) -> JSONType:

        width = int(self.img_width.value)
        height = int(self.img_height.value)

        parameters_input = {
            "connector": "image",
            "width": width,
            "height": height,
            "bw": self.bw.value,
            "histogram_equalization": self.histogram_equalization.value,
            "db": True,
        }

        if self.unchanged_data.value:
            parameters_input["unchanged_data"] = True

        if self.multi_label.value:
            parameters_input["multi_label"] = True
            parameters_input["db"] = False

        if self.ctc.value:
            parameters_input["ctc"] = True

        if self.rgb.value:
            parameters_input["rgb"] = self.rgb.value

        if self.scale.value:
            parameters_input["scale"] = self.scale.value
            
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


        dic["engine"] = self.engine.value

        if self.mllib.value == 'caffe':
            crop_size = int(self.crop_size.value)
            if crop_size > 0:
                dic["crop_size"] = crop_size
            #if self.noise_prob.value > 0.0:
            dic["noise"] = {"all_effects": True, "prob": self.noise_prob.value}
            #if self.distort_prob.value > 0.0:
            dic["distort"] = {
                    "all_effects": True,
                    "prob": self.distort_prob.value,
                }
            if not hasattr(self,'persp_horizontal'):
                pass
            else:
                if any(
                        [
                            #self.all_effects.value,
                            self.persp_horizontal.value,
                            self.persp_vertical.value,
                            self.zoom_out.value,
                            self.zoom_in.value,
                        ]
                ) or any(
                    p != ""
                    for p in [
                            self.persp_factor.value,
                            self.zoom_factor.value,
                            self.pad_mode.value,
                            self.geometry_prob.value,
                    ]
                ):
                    dic["geometry"] = {}
                    # -- booleans --
                    #if self.all_effects.value:
                    #    dic["geometry"]["all_effects"] = True
                    if self.persp_horizontal.value:
                        dic["geometry"]["persp_horizontal"] = True
                    if self.persp_vertical.value:
                        dic["geometry"]["persp_vertical"] = True
                    if self.zoom_out.value:
                        dic["geometry"]["zoom_out"] = True
                    if self.zoom_in.value:
                        dic["geometry"]["zoom_in"] = True
                    # -- strings --
                    if self.pad_mode.value != "":
                        dic["geometry"]["pad_mode"] = self.pad_mode.value
                        # -- float --
                    if self.persp_factor.value != "":
                        dic["geometry"]["persp_factor"] = float(self.persp_factor.value)
                    if self.zoom_factor.value != "":
                        dic["geometry"]["zoom_factor"] = float(self.zoom_factor.value)
                    if self.geometry_prob.value != "":
                        dic["geometry"]["prob"] = float(self.geometry_prob.value)

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
                "warmup_lr": self.warmup_lr.value,
                "warmup_iter": self.warmup_iter.value,
                "solver_type": self.solver_type.value,
                "sam" : self.sam.value,
                "swa" : self.swa.value,
                "lookahead": self.lookahead.value,
                "lookahead_steps": self.lookahead_steps.value,
                "lookahead_alpha": self.lookahead_alpha.value,
                "rectified": self.rectified.value,
                "decoupled_wd_periods": self.decoupled_wd_periods.value,
                "decoupled_wd_mult": self.decoupled_wd_mult.value,
                "lr_dropout": self.lr_dropout.value,
                "iter_size": self.iter_size.value,
                "lr_policy": self.lr_policy.value,
            },
            "engine": self.engine.value,
        }

        if self.mllib.value == 'torch':
            dic["rotate"] = self.rotate.value
            dic["mirror"] = self.mirror.value
            crop_size = int(self.crop_size.value)
            if crop_size > 0:
                dic["crop_size"] = crop_size
            dic["noise"] = {"prob": self.noise_prob.value}
            dic["distort"] = {
                    "prob": self.distort_prob.value
                }
            dic["cutout"] = self.cutout_prob.value
            if not hasattr(self,'persp_horizontal'):
                pass
            else:
                if any(
                        [
                            self.persp_horizontal.value,
                            self.persp_vertical.value,
                            self.transl_horizontal.value,
                            self.transl_vertical.value,
                            self.zoom_out.value,
                            self.zoom_in.value,
                        ]
                ) or any(
                    p != ""
                    for p in [
                            self.persp_factor.value,
                            self.transl_factor.value,
                            self.zoom_factor.value,
                            self.pad_mode.value,
                            self.geometry_prob.value,
                    ]
                ):
                    dic["geometry"] = {}
                    # -- booleans --
                    if self.persp_horizontal.value:
                        dic["geometry"]["persp_horizontal"] = True
                    if self.persp_vertical.value:
                        dic["geometry"]["persp_vertical"] = True
                    if self.transl_horizontal.value:
                        dic["geometry"]["transl_horizontal"] = True
                    if self.transl_vertical.value:
                        dic["geometry"]["transl_vertical"] = True
                    if self.zoom_out.value:
                        dic["geometry"]["zoom_out"] = True
                    if self.zoom_in.value:
                        dic["geometry"]["zoom_in"] = True
                    # -- strings --
                    if self.pad_mode.value != "":
                        dic["geometry"]["pad_mode"] = self.pad_mode.value
                        # -- float --
                    if self.persp_factor.value != "":
                        dic["geometry"]["persp_factor"] = float(self.persp_factor.value)
                    if self.transl_factor.value != "":
                        dic["geometry"]["transl_factor"] = float(self.transl_factor.value)
                    if self.zoom_factor.value != "":
                        dic["geometry"]["zoom_factor"] = float(self.zoom_factor.value)
                    if self.geometry_prob.value != "":
                        dic["geometry"]["prob"] = float(self.geometry_prob.value)
        
        if self.stepvalue.value:
            dic["solver"]["stepvalue"] = eval(self.stepvalue.value)
        
        if self.rand_skip.value > 0 and self.resume.value:
            dic["solver"]["rand_skip"] = self.rand_skip.value
        if self.class_weights.value:
            dic["class_weights"] = eval(self.class_weights.value)
        if self.ignore_label.value >= 0:
            dic["ignore_label"] = int(self.ignore_label.value)

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
