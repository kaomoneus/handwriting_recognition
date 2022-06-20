import argparse
from pathlib import Path
from typing import Tuple

from keras.saving.save import load_model

from config import CACHE_DIR_DEFAULT
from dataset_utils import GroundTruthPathsItem, ROI, preprocess_dataset, load_dataset
from plot_utils import tf_plot_predictions, plot_dataset, plot_interactive
from text_utils import load_vocabulary, add_voc_args, parse_voc_args, Vocabulary


def register_ploti_args(ploti_cmd: argparse.ArgumentParser):
    ploti_cmd.add_argument("-img", help="Root directory with images", required=True)
    ploti_cmd.add_argument("-text", help="File with text ground truth", required=True)

    add_voc_args(ploti_cmd)


def handle_ploti_cmd(args: argparse.Namespace):
    run_ploti(
        img_path=args.img,
        text_path=args.text,
        vocabulary=parse_voc_args(args)
    )


def run_ploti(img_path: str, text_path: str, vocabulary: Vocabulary):

    dataset, _ = load_dataset(
        str_values_file_path=text_path,
        img_dir=img_path,
        vocabulary=vocabulary,
    )

    plot_interactive(dataset, 5, 5)
