import logging
from pathlib import Path
from tempfile import mkstemp
from typing import Iterator, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio, Image, display

import cv2
import librosa
from tqdm.autonotebook import tqdm

from .mixins import ImageTrainerMixin
from .widgets import GPUIndex, Solver


def make_slice(total: int, size: int, step: int) -> Iterator[slice]:
    """
    Sliding window over the melody. step should be less than or equal to size.
    """

    if step > size:
        logging.warn("step > size, you probably miss some part of the melody")
    if total < size:
        yield slice(0, total)
        return
    for t in range(0, total - size, step):
        yield slice(t, t + size)
    if t + size < total:
        yield slice(total - size, total)


def build_dir(src_dir: Path, dst_dir: Path):

    if not dst_dir.exists():
        dst_dir.mkdir()

    for directory in src_dir.glob("*"):

        new_dir = dst_dir / directory.stem
        if not new_dir.exists():
            new_dir.mkdir()

        # build the list first to get its size...
        file_list = list(directory.glob("*"))
        for file in tqdm(file_list, desc=directory.name):
            f = file.relative_to(src_dir)
            # do not open the file (long) if the image already exists!
            if sum(1 for _ in new_dir.glob(f"{f.stem}_*.exr")) == 0:
            # if not (new_dir / f"{f.stem}_00000_00257.exr").exists():
                y, sr = librosa.load(file)
                # 2^9 seems a good compromise, maybe pass it as a parameter in
                # the future.
                D = librosa.stft(y, 2 ** 9, center=True)
                spec = librosa.amplitude_to_db(
                    librosa.magphase(D)[0], ref=np.max
                )
                for slc in make_slice(spec.shape[1], 257, 100):
                    pattern = f"{f.stem}_{slc.start:>05d}_{slc.stop:>05d}.exr"
                    cv2.imwrite((new_dir / pattern).as_posix(), spec[:, slc])


class AudioClassification(ImageTrainerMixin):

    def __init__(  # type: ignore
        self,
        sname: str,
        *,  # unnamed parameters are forbidden
        mllib: str = "caffe",
        training_repo: Path = None,
        testing_repo: Path = None,
        tmp_dir: Path = None,
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
        lookahead : bool = False,
        lookahead_steps : int = 6,
        lookahead_alpha : float = 0.5,
        rectified : bool = False,
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
        unchanged_data: bool = False,
        ctc: bool = False,
        target_repository: str = "",
        **kwargs
    ) -> None:

        super().__init__(sname, locals())

    def _train_service_body(self):
        body = super()._train_service_body()

        tmp_dir = Path(self.tmp_dir.value)
        train_dir = Path(self.training_repo.value)
        test_dir = Path(self.testing_repo.value)
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)

        build_dir(train_dir, tmp_dir / "train")
        body['data'] = [(tmp_dir / "train").as_posix()]

        if self.testing_repo.value != "":
            build_dir(test_dir, tmp_dir / "test")
            body['data'] += [(tmp_dir / "test").as_posix()]

        return body

    def display_img(self, args):
        self.output.clear_output()
        with self.output:
            for filepath in args["new"]:
                display(Audio(filepath, autoplay=True))

                y, sr = librosa.load(filepath)
                D = librosa.stft(y, 2 ** 9, center=True)
                spec = librosa.amplitude_to_db(
                    librosa.magphase(D)[0], ref=np.max
                )

                fig, ax = plt.subplots(1, 5, figsize=(16, 4))

                for i, sl in zip(range(5), make_slice(spec.shape[1], 257, 100)):
                    ax[i].imshow(spec[:, sl])

                _, fname = mkstemp(suffix=".png")
                fig.savefig(fname)

                display(Image(fname))
