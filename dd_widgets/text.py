from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

from ipywidgets import HBox, SelectMultiple

from .core import sample_from_iterable
from .widgets import MLWidget, Solver, GPUIndex

alpha = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:’\“/\_@#$%^&*~`+-=<>()[]{}"


class Text(MLWidget):
    def __init__(
        self,
        sname: str,
        *,
        mllib: str = "caffe",
        training_repo: Path,
        testing_repo: Optional[Path] = None,
        description: str = "Text service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        gpuid: GPUIndex = 0,
        path: str = "",
        regression: bool = False,
        db: bool = True,
        nclasses: int = -1,
        ignore_label: Optional[int] = -1,
        layers: List[str] = [],
        iterations: int = 25000,
        test_interval: int = 1000,
        base_lr: float = 0.001,
        resume: bool = False,
        solver_type: Solver = "SGD",
        batch_size: int = 128,
        shuffle: bool = True,
        tsplit: float = 0.2,
        min_count: int = 10,
        min_word_length: int = 5,
        count: bool = False,
        tfidf: bool = False,
        sentences: bool = False,
        characters: bool = False,
        sequence: int = -1,
        read_forward: bool = True,
        alphabet: str = alpha,
        sparse: bool = False,
        template: Optional[str] = None,
        activation: str = "relu",
        embedding: bool = False,
        target_repository: str = ""
    ) -> None:

        super().__init__(sname, locals())

        self.train_labels = SelectMultiple(
            options=[], value=[], description="Training labels", disabled=False
        )

        self.test_labels = SelectMultiple(
            options=[], value=[], description="Testing labels", disabled=False
        )

        # self.testing_repo.observe(self.update_label_list, names="value")
        self.training_repo.observe(  # type: ignore
            self.update_label_list, names="value"
        )

        self.train_labels.observe(self.update_train_file_list, names="value")
        self.test_labels.observe(self.update_test_file_list, names="value")
        self.file_list.observe(self.display_text, names="value")

        self.update_label_list(())

        self._img_explorer.children = [
            HBox([HBox([self.train_labels, self.test_labels])]),
            self.file_list,
            self.output,
        ]

        if self.characters:  # type: ignore
            self.db.value = True  # type: ignore

    def display_text(self, args):
        self.output.clear_output()
        with self.output:
            for path in args["new"]:
                with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                    for i, x in enumerate(fh.readlines()):
                        if i == 20:
                            break
                        print(x.strip())

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

    def _create_service_body(self):

        body = OrderedDict(
            [
                ("mllib", self.mllib.value),
                ("description", "text classification service"),
                ("type", "supervised"),
                (
                    "parameters",
                    {
                        "input": {
                            "connector": "txt",
                            "characters": self.characters.value,
                            "sequence": self.sequence.value,
                            "read_forward": self.read_forward.value,
                            "alphabet": self.alphabet.value,
                            "sparse": self.sparse.value,
                            "embedding": self.embedding.value,
                        },
                        "mllib": {
                            "template": self.template.value,
                            "nclasses": self.nclasses.value,
                            "layers": eval(self.layers.value),
                            "activation": self.activation.value,
                            "db": self.db.value,
                            "regression": self.regression.value,
                        },
                        "output": {"store_config": True},
                    },
                ),
                (
                    "model",
                    {
                        "templates": "../templates/caffe/",
                        "repository": self.model_repo.value,
                        "create_repository": True,
                    },
                ),
            ]
        )

        if self.regression.value:
            del body["parameters"]["mllib"]["nclasses"]
            body["parameters"]["mllib"]["ntargets"] = int(self.ntargets.value)

        return body

    def _train_body(self):
        assert len(self.gpuid.index) > 0, "Set a GPU index"
        body = OrderedDict(
            [
                ("service", self.sname),
                ("async", True),
                (
                    "parameters",
                    {
                        "mllib": {
                            "gpu": True,
                            "resume": self.resume.value,
                            "gpuid": (
                                list(self.gpuid.index)
                                if len(self.gpuid.index) > 1
                                else self.gpuid.index[0]
                            ),
                            "solver": {
                                "iterations": self.iterations.value,
                                "test_interval": self.test_interval.value,
                                "test_initialization": False,
                                "base_lr": self.base_lr.value,
                                "solver_type": self.solver_type.value,
                            },
                            "net": {"batch_size": self.batch_size.value},
                        },
                        "input": {
                            "shuffle": self.shuffle.value,
                            "test_split": self.tsplit.value,
                            "min_count": self.min_count.value,
                            "min_word_length": self.min_word_length.value,
                            "count": self.count.value,
                            "tfidf": self.tfidf.value,
                            "sentences": self.sentences.value,
                            "characters": self.characters.value,
                            "sequence": self.sequence.value,
                            "read_forward": self.read_forward.value,
                            "alphabet": self.alphabet.value,
                            "embedding": self.embedding.value,
                            "db": self.db.value,
                        },
                        "output": {"measure": ["mcll", "f1", "cmdiag"]},
                    },
                ),
                ("data", [self.training_repo.value]),
            ]
        )

        if self.mllib.value == "xgboost":
            del body["parameters"]["mllib"]["solver"]
            body["parameters"]["mllib"]["iterations"] = self.iterations.value

        if self.ignore_label.value != -1:
            body["parameters"]["mllib"]["ignore_label"] = int(
                self.ignore_label.value
            )

        return body
