import argparse
from pathlib import Path
from typing import Tuple

from keras.saving.save import load_model

from config import CACHE_DIR_DEFAULT
from dataset_utils import GroundTruthPathsItem, ROI, preprocess_dataset
from plot_utils import tf_plot_predictions, plot_dataset
from text_utils import load_vocabulary


def register_plot_args(recognize_cmd: argparse.ArgumentParser):
    recognize_cmd.add_argument("-img", help="Path to image to be processed", required=True)

    # TODO: Make it possible to pass several roi instances
    #    in this case we will able to recognize several fields in given image.
    recognize_cmd.add_argument(
        "-roi", help="Region of interest: <x, y, width, height>",
        type=lambda roi_str: tuple(map(int, roi_str.replace(" ", "").split(",")))
    )
    recognize_cmd.add_argument(
        "-keep", dest="keep",
        action="store_true",
        help="Use cached files (don't overwrite them)"
    )


def handle_plot_cmd(args: argparse.Namespace):
    run_plot(
        img_path=args.img,
        roi=args.roi,
        keep=args.keep,
    )


def run_plot(img_path: str, roi: ROI, keep: bool):

    ds = [GroundTruthPathsItem(
        str_value="",
        img_path=img_path,
        img_name=Path(img_path).stem,
        roi=roi
    )]

    ds = preprocess_dataset(
        ds,
        only_threshold=False, cache_dir=CACHE_DIR_DEFAULT,
        keep=keep,
        full=False
    )

    plot_dataset(ds)


