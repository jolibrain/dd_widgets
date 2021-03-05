from pathlib import Path
from typing import List, Optional

from ipywidgets import HBox, SelectMultiple

from .core import JSONType
from .mixins import TextTrainerMixin, sample_from_iterable
from .widgets import Solver, GPUIndex, Engine

alpha = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:’\“/\_@#$%^&*~`+-=<>()[]{}"


class Text(TextTrainerMixin):
    def __init__(
        self,
        sname: str,
        *,
        mllib: str = "caffe",
        engine: Engine = "CUDNN_SINGLE_HANDLE",
        training_repo: Path,
        testing_repo: Optional[Path] = None,
        description: str = "Text service",
        model_repo: Path = None,
        host: str = "localhost",
        port: int = 1234,
        path: str = "",
        gpuid: GPUIndex = 0,
        # -- specific
        regression: bool = False,
        db: bool = True,
        nclasses: int = -1,
        ignore_label: Optional[int] = -1,
        layers: List[str] = [],
        iterations: int = 25000,
        test_interval: int = 1000,
        snapshot_interval: int = 1000,
        base_lr: float = 0.001,
        lr_policy: str = "fixed",
        stepvalue: List[int] = [],
        warmup_lr: float = 0.0001,
        warmup_iter: int = 0,
        resume: bool = False,
        solver_type: Solver = "SGD",
        sam : bool = False,
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        rectified : bool = False,
        decoupled_wd_periods : int = 4,
        decoupled_wd_mult : float = 2.0,
        lr_dropout : float = 1.0,
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
        autoencoder: bool = False,
        lregression: bool = False,
        dropout: float = .2,
        finetune: bool = False,
        weights: str = "",
        class_weights: List[float] = [],
        test_batch_size: int = 16,
        iter_size: int = 1,
        target_repository: str = "",
        ##-- new txt input conns stuff for bert and gpt2
        ordered_words: bool = True,
        wordpiece_tokens: bool = True,
        punctuation_tokens: bool = True,
        lower_case: bool =False,
        word_start: str = "Ġ",
        suffix_start: str = "",
        ##--end bert, gpt2 new stuff
        embedding_size: int = 768,
        freeze_traced: bool = False,
        **kwargs
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

        if self.mllib.value == "torch":
            self.db.value = False
            
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

    def _create_parameters_input(self) -> JSONType:
        return {
            "connector": "txt",
            "characters": self.characters.value,
            "sequence": self.sequence.value,
            "read_forward": self.read_forward.value,
            "alphabet": self.alphabet.value,
            "sparse": self.sparse.value,
            "embedding": self.embedding.value,
            "ordered_words": self.ordered_words.value,
            "wordpiece_tokens": self.wordpiece_tokens.value,
            "punctuation_tokens": self.punctuation_tokens.value,
            "lower_case": self.lower_case.value,
            "word_start": self.word_start.value,
            "suffix_start": self.suffix_start.value,
        }

    def _create_parameters_mllib(self) -> JSONType:
        dic = super()._create_parameters_mllib()
        dic["embedding_size"] = self.embedding_size.value
        dic["freeze_traced"] = self.freeze_traced.value
        return dic
    
    def _train_parameters_input(self) -> JSONType:
        return {
            "alphabet": self.alphabet.value,
            "characters": self.characters.value,
            "count": self.count.value,
            "db": self.db.value,
            "embedding": self.embedding.value,
            "min_count": self.min_count.value,
            "min_word_length": self.min_word_length.value,
            "read_forward": self.read_forward.value,
            "sentences": self.sentences.value,
            "sequence": self.sequence.value,
            "shuffle": self.shuffle.value,
            "test_split": self.tsplit.value,
            "tfidf": self.tfidf.value,
        }
