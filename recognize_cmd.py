import argparse
from pathlib import Path
from typing import Tuple

from keras.saving.save import load_model

from config import CACHE_DIR_DEFAULT
from dataset_utils import ROI, preprocess_dataset, tf_dataset
from common import GroundTruthPathsItem
from plot_utils import tf_plot_predictions
from text_utils import load_vocabulary, add_voc_args, parse_voc_args, Vocabulary


def register_recognize_args(recognize_cmd: argparse.ArgumentParser):
    recognize_cmd.add_argument("-img", help="Path to image to be recognized", required=True)
    recognize_cmd.add_argument("-model", help="Path to NN model", required=True)

    # TODO: Make it possible to pass several roi instances
    #    in this case we will able to recognize several fields in given image.
    recognize_cmd.add_argument(
        "-roi",
        help="Region of interest: <x, y, width, height>",
        type=lambda roi_str: tuple(map(int, roi_str.replace(" ", "").split(",")))
    )
    recognize_cmd.add_argument(
        "-preprocess",
        help="Also apply preprocessing",
        action="store_true"
    )

    add_voc_args(recognize_cmd)


def handle_recognize_cmd(args: argparse.Namespace):
    run_recognize(
        img_path=args.img,
        model_path=args.model,
        roi=args.roi,
        preprocess=args.preprocess,
        vocabulary=parse_voc_args(args),
    )


def run_recognize(
    img_path: str,
    model_path: str,
    roi: ROI,
    preprocess: bool,
    vocabulary: Vocabulary,
):

    ds = [GroundTruthPathsItem(
        str_value="",
        img_path=img_path,
        roi=roi,
        img_name=Path(img_path).stem
    )]

    ds = preprocess_dataset(
        ds,
        only_threshold=(not preprocess),
        full=preprocess,
        cache_dir=CACHE_DIR_DEFAULT
    )

    model = load_model(model_path)

    tf_plot_predictions(model, tf_dataset(ds, vocabulary), vocabulary)
