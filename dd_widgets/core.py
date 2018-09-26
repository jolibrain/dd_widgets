import json
import logging
import os
import random
import shutil
from collections import OrderedDict
from heapq import nlargest
from pathlib import Path
from tempfile import mkstemp
from typing import Iterator, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
from IPython.display import Image
from matplotlib import patches
from matplotlib.cm import get_cmap

import cv2


class ImageTrainerMixin:
    def _create_service_body(self):
        width = int(self.img_width.value)
        height = int(self.img_height.value)
        crop_size = int(self.crop_size.value)

        nclasses = int(self.nclasses.value)
        if nclasses == -1:
            nclasses = len(os.walk(self.training_repo.value).next()[1])

        logging.info("{} classes".format(nclasses))
        description = self.description.value
        mllib = "caffe"

        model = {
            "templates": "../templates/caffe/",
            "repository": self.model_repo.value,
            "create_repository": True,
        }

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

        if self.__class__.__name__ == "Detection":
            parameters_input["bbox"] = True

        if self.__class__.__name__ == "Segmentation":
            parameters_input["segmentation"] = True

        if self.multi_label.value:
            parameters_input["multi_label"] = True
            parameters_input["db"] = False
        if self.ctc.value:
            parameters_input["ctc"] = True

        logging.info(
            "Parameters input: {}".format(
                json.dumps(parameters_input, indent=2)
            )
        )

        if not self.finetune.value:
            if self.template.value:
                parameters_mllib = {
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
                parameters_mllib = {
                    "nclasses": nclasses,
                    "mirror": self.mirror.value,
                    "rotate": self.rotate.value,
                    "scale": self.scale.value,
                    "autoencoder": self.autoencoder.value,
                }
        else:
            if self.template.value:
                parameters_mllib = {
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
                parameters_mllib = {
                    "finetuning": True,
                    "nclasses": nclasses,
                    "weights": self.weights.value,
                    "rotate": self.rotate.value,
                    "mirror": self.mirror.value,
                    "scale": self.scale.value,
                    "autoencoder": self.autoencoder.value,
                }
        if self.multi_label.value:
            parameters_mllib["db"] = False

        if crop_size > 0:
            parameters_mllib["crop_size"] = crop_size
        if self.noise_prob.value > 0.0:
            parameters_mllib["noise"] = {
                "all_effects": True,
                "prob": self.noise_prob.value,
            }
        if self.distort_prob.value > 0.0:
            parameters_mllib["distort"] = {
                "all_effects": True,
                "prob": self.distort_prob.value,
            }
        parameters_mllib["gpu"] = True
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        parameters_mllib["gpuid"] = (
            list(self.gpuid.index)
            if len(self.gpuid.index) > 1
            else self.gpuid.index[0]
        )
        if self.regression.value:
            parameters_mllib["regression"] = True

        if self.__class__.__name__ == "Segmentation":
            parameters_mllib["loss"] = self.loss.value

        logging.info(
            "Parameters mllib: {}".format(
                json.dumps(parameters_input, indent=2)
            )
        )

        parameters_output = {'store_config':True}
        # print (parameters_input)
        # print (parameters_mllib)
        # pserv = dd.put_service(self.sname.value,model,description,mllib,
        #                       parameters_input,parameters_mllib,parameters_output)

        body = OrderedDict(
            [  # typing: Dict[str, Any]
                ("description", description),
                ("mllib", mllib),
                ("type", "supervised"),
                (
                    "parameters",
                    {
                        "input": parameters_input,
                        "mllib": parameters_mllib,
                        "output": parameters_output,
                    },
                ),
                ("model", model),
            ]
        )
        return body

    def _train_body(self):

        train_data = [self.training_repo.value]
        parameters_input = {
            "test_split": self.tsplit.value,
            "shuffle": True,
            "db": True,
        }

        if self.__class__.__name__ == "Segmentation":
            parameters_input["segmentation"] = True

        if self.__class__.__name__ == "Detection":
            parameters_input["db_width"] = self.db_width.value
            parameters_input["db_height"] = self.db_height.value

        if self.testing_repo.value != "":
            train_data.append(self.testing_repo.value)
            parameters_input["shuffle"] = True

        if self.multi_label.value:
            parameters_input["db"] = False

        if self.ctc.value:
            if self.align.value:
                parameters_input["align"] = True
                
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        parameters_mllib = {
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
            parameters_mllib["bbox"] = True

        # TODO: lr policy as arguments
        # 'lr_policy':'step','stepsize':2000,'gamma':0.1,'snapshot':4000,'base_lr':args.base_lr,'solver_type':'SGD'}}
        if self.rand_skip.value > 0 and self.resume.value:
            parameters_mllib["solver"]["rand_skip"] = self.rand_skip.value
        if self.class_weights.value:
            parameters_mllib["class_weights"] = eval(self.class_weights.value)
        if self.ignore_label.value >= 0:
            parameters_mllib["ignore_label"] = self.ignore_label.value
        if self.timesteps.value:
            parameters_mllib["timesteps"] = self.timesteps.value

        if self.__class__.__name__ == "Classification":
            parameters_output = {"measure": ["mcll", "f1", "acc-5"]}
        elif self.__class__.__name__ == "Segmentation":
            parameters_output = {"measure": ["acc"]}
        elif self.__class__.__name__ == "Detection":
            parameters_output = {"measure": ["map"]}
        elif self.__class__.__name__ == "OCR":
            parameters_output = {"measure": ["acc"]}
        elif self.multi_label.value and self.regression.value:
            parameters_output = {
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
        elif self.ctc:
            parameters_output = {"measure": ["acc"]}
        else:
            parameters_output = {"measure": ["mcll", "f1", "acc-5"]}

        parameters_output["target_repository"] = ""

        body = OrderedDict(
            [
                ("service", self.sname),
                ("async", True),
                (
                    "parameters",
                    {
                        "input": parameters_input,
                        "mllib": parameters_mllib,
                        "output": parameters_output,
                    },
                ),
                ("data", train_data),
            ]
        )

        return body


Elt = TypeVar("Elt")


def sample_from_iterable(it: Iterator[Elt], k: int) -> Iterator[Elt]:
    return (x for _, x in nlargest(k, ((random.random(), x) for x in it)))


def img_handle(
    path: Path,
    segmentation: Optional[Path] = None,
    bbox: Optional[Path] = None,
    nclasses: int = -1
) -> Tuple[Tuple[int, ...], Image]:
    data = cv2.imread(path.as_posix(), cv2.IMREAD_UNCHANGED)
    _, fname = mkstemp(suffix=".png")
    fig, ax = plt.subplots()
    ax.imshow(data)
    if segmentation is not None:
        data = cv2.imread(segmentation.as_posix(), cv2.IMREAD_UNCHANGED)
        ax.imshow(data, alpha=.8)
        if data.max() >= nclasses > -1:
            raise RuntimeError(
                "Index {max} present in {filename}".format(
                    max=data.max(),
                    filename=segmentation.as_posix()
                )
            )
    if bbox is not None:
        
        if nclasses > -1:
            cmap = get_cmap('jet', nclasses-1)
            
        with bbox.open("r") as fh:
            for line in fh.readlines():
                tag, xmin, ymin, xmax, ymax = (
                    int(float(x)) for x in line.strip().split()
                )
                if tag >= nclasses > -1:
                    raise RuntimeError(
                        "Index {max} present in {filename}".format(
                            max=tag,
                            filename=bbox.as_posix()
                        )
                    )                    
                rect = patches.Rectangle(
                    (xmin, ymin),
                    xmax - xmin,
                    ymax - ymin,
                    linewidth=2,
                    edgecolor=cmap(tag) if nclasses > -1 else 'blue',
                    facecolor="none",
                )
                ax.add_patch(rect)

    fig.savefig(fname)
    plt.close(fig)
    return data.shape, Image(fname)
