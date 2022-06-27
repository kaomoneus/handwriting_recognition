import argparse
import logging

import numpy as np
from keras.saving.save import load_model
from tqdm import tqdm

from config import CACHE_DIR_DEFAULT
from dataset_utils import add_dataset_args, parse_dataset_args, tf_dataset, preprocess_dataset
from model_utils import prediction_model, calculate_edit_distance
from plot_utils import tf_plot_predictions
from text_utils import add_voc_args, parse_voc_args
import tensorflow as tf


LOG = logging.getLogger()


def register_eval_cmd(eval_cmd: argparse.ArgumentParser):
    eval_cmd.add_argument(
        "-i", dest="initial_model",
        required=True,
        help="Initial model path. If provided then saved model will be loaded."
    )
    # TODO: Consider making it part of dataset args set.
    eval_cmd.add_argument(
        "-preprocess",
        action="store_true",
        help="Preprocess input tasks"
    )
    add_voc_args(eval_cmd)
    add_dataset_args(eval_cmd)


def handle_eval_cmd(args: argparse.Namespace):
    vocabulary = parse_voc_args(args)
    dataset, _ = parse_dataset_args(args, vocabulary)

    if args.preprocess:
        dataset = preprocess_dataset(
            dataset,
            cache_dir=CACHE_DIR_DEFAULT,
            only_threshold=False,
        )

    model = load_model(args.initial_model)
    predictor = prediction_model(model)

    ds_len = len(dataset)
    dataset = tf_dataset(dataset, vocabulary, shuffle=False, batch_size=1)

    edit_distances = []

    for batch in tqdm(dataset, desc=f"Evaluating"):
        images = batch["image"]
        labels = batch["label"]

        predictions = predictor.predict(images, verbose=0)

        edit_distances.append(
            calculate_edit_distance(labels, predictions, vocabulary).numpy(),
        )

    med = np.mean(edit_distances)
    LOG.info(f"Mean edit distance: {med:.4f}")

    # TODO: use interactive ploting
    tf_plot_predictions(model, dataset, vocabulary)
