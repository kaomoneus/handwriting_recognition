import argparse
from pathlib import Path
from typing import Tuple

from keras.saving.save import load_model

from dataset_utils import GroundTruthItem, tf_dataset
from image_utils import load_and_pad_image
from model_utils import prediction_model
from plot_utils import plot_predictions
from text_utils import load_vocabulary


def register_recognize_args(recognize_cmd: argparse.ArgumentParser):
    recognize_cmd.add_argument("-img", help="Path to image to be recognized", required=True)
    recognize_cmd.add_argument("-model", help="Path to NN model", required=True)
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


def run_recognize(img_path: str, model_path: str, roi: Tuple[int, int, int, int]):
    ds = [GroundTruthItem(
        str_value="",
        img=load_and_pad_image(image_path=img_path, roi=roi)
    )]

    vocabulary = load_vocabulary(str(Path(model_path).with_suffix(".voc")))
    model = load_model(model_path)

    plot_predictions(model, ds, vocabulary)


