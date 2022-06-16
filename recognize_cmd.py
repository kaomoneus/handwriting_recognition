import argparse
from pathlib import Path
from typing import Tuple

from keras.saving.save import load_model

from config import CACHE_DIR_DEFAULT
from dataset_utils import GroundTruthPathsItem, ROI, preprocess_dataset
from plot_utils import plot_predictions
from text_utils import load_vocabulary


def register_recognize_args(recognize_cmd: argparse.ArgumentParser):
    recognize_cmd.add_argument("-img", help="Path to image to be recognized", required=True)
    recognize_cmd.add_argument("-model", help="Path to NN model", required=True)

    # TODO: Make it possible to pass several roi instances
    #    in this case we will able to recognize several fields in given image.
    recognize_cmd.add_argument(
        "-roi", help="Region of interest: <x, y, width, height>",
        type=lambda roi_str: tuple(map(int, roi_str.replace(" ", "").split(",")))
    )


def handle_recognize_cmd(args: argparse.Namespace):
    run_recognize(
        img_path=args.img,
        model_path=args.model,
        roi=args.roi
    )


def run_recognize(img_path: str, model_path: str, roi: ROI):

    ds = [GroundTruthPathsItem(
        str_value="",
        img_path=img_path,
        roi=roi,
        img_name=Path(img_path).stem
    )]

    ds = preprocess_dataset(ds, only_threshold=True, cache_dir=CACHE_DIR_DEFAULT)

    vocabulary = load_vocabulary(str(Path(model_path).with_suffix(".voc")))
    model = load_model(model_path)

    plot_predictions(model, ds, vocabulary)


