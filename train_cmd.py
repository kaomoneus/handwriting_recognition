import argparse
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from keras.backend import clear_session
from keras.saving.save import load_model
from tensorflow import keras
from tqdm import tqdm

from config import CACHE_DIR_DEFAULT, TRAIN_EPOCHS_DEFAULT, TRAIN_TEST_RATIO, TRAIN_VALIDATE_CNT, DATASET_SHUFFLER_SEED
from dataset_utils import Dataset, tf_dataset, load_dataset, preprocess_dataset, load_marked
from model_utils import build_model, prediction_model
from plot_utils import tf_plot_samples, tf_plot_predictions
from text_utils import Vocabulary, add_voc_args, parse_voc_args


LOG = logging.getLogger(__name__)


def register_train_args(train_cmd: argparse.ArgumentParser):
    train_cmd.add_argument(
        "-text",
        required=True,
        help="ASCII file with lines or sentences description.",
    )
    train_cmd.add_argument(
        "-img",
        required=True,
        help="Root directory of IAM image lines or sentences files",
    )
    train_cmd.add_argument(
        "-i", dest="initial_model",
        help="Initial model path. If provided then saved model will be loaded."
    )
    train_cmd.add_argument(
        "-o", dest="output_path",
        required=True,
        help="Output model path. Model will be trained and saved at this path."
    )
    train_cmd.add_argument(
        "-e", dest="epochs",
        default=TRAIN_EPOCHS_DEFAULT,
        type=int,
        help="Number of train epochs."
    )
    train_cmd.add_argument(
        "-validation-list", dest="validation_list",
        help="Path to file to store list of validation samples"
    )
    train_cmd.add_argument(
        "-plot", dest="plot",
        action="store_true",
        help="Plot samples and predictions"
    )
    train_cmd.add_argument(
        "-max-ds-items",
        type=int,
        help="Maximum amount of loaded datasource items"
    )
    train_cmd.add_argument(
        "-blacklist",
        help="File with blacklisted item names. Should be in format of 'ploti' state."
    )

    add_voc_args(train_cmd)


def handle_train_cmd(args: argparse.Namespace):
    run_train(
        text_path=args.text,
        img_root_path=args.img,
        initial_model_path=args.initial_model,
        output_model=args.output_path,
        epochs=args.epochs,
        validation_list=args.validation_list,
        plot=args.plot,
        vocabulary=parse_voc_args(args),
        max_ds_items=args.max_ds_items,
        blacklist_path=args.blacklist,
    )


def init_model(model_path: str, vocabulary: Vocabulary) -> keras.Model:
    if not model_path:
        return build_model(vocabulary)

    return load_model(model_path)


def train_model(
    model: keras.Model,
    epochs: int,
    train_ds: Dataset, validate_ds: Dataset,
    vocabulary: Vocabulary,
    plot: bool,
):
    tf_train_ds = tf_dataset(train_ds, vocabulary, resize=False)
    tf_validation_ds = tf_dataset(validate_ds, vocabulary, resize=False)

    if plot:
        tf_plot_samples(tf_train_ds, vocabulary)

    validation_images = []
    validation_labels = []

    for batch in tf_validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    validation_set_size = len(validation_images)

    def calculate_edit_distance(labels, predictions, vocabulary: Vocabulary):
        # Get a single batch and convert its labels to sparse tensors.
        saprse_labels = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

        # Make predictions and convert them to sparse tensors.
        input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
        predictions_decoded = tf.keras.backend.ctc_decode(
            predictions, input_length=input_len, greedy=True
        )[0][0][:, :vocabulary.max_len]
        sparse_predictions = tf.cast(
            tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
        )

        # Compute individual edit distances and average them out.
        edit_distances = tf.edit_distance(
            sparse_predictions, saprse_labels, normalize=False
        )
        return tf.reduce_mean(edit_distances)

    class EditDistanceCallback(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.prediction_model = prediction_model(model)

        def on_epoch_end(self, epoch, logs=None):
            edit_distances = []

            # FIXME: Evaluation is slow, may be because of 'tqdm' I don't know...
            for i in tqdm(range(validation_set_size), desc=f"Evaluating epoch #{epoch}"):
                labels = validation_labels[i]
                predictions = self.prediction_model.predict(
                    validation_images[i],
                    verbose=0
                )
                edit_distances.append(calculate_edit_distance(labels, predictions, vocabulary).numpy())
                clear_session()

            print(
                f"Mean edit distance for epoch {epoch + 1}: {np.mean(edit_distances):.4f}"
            )

    edit_distance_callback = EditDistanceCallback()

    # Train the model.
    history = model.fit(
        tf_train_ds,
        validation_data=tf_validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback],
    )

    return history


def run_train(
    text_path,
    img_root_path,
    initial_model_path,
    epochs,
    output_model,
    validation_list,
    plot,
    vocabulary: Optional[Vocabulary],
    max_ds_items: int,
    blacklist_path: Optional[str]
):

    blacklist = None
    if blacklist_path:
        state = load_marked(blacklist_path)
        blacklist = set(state.marked)
        LOG.info(f"Loaded blacklist with {len(blacklist)} items.")

    dataset, auto_voc = load_dataset(
        str_values_file_path=text_path, img_dir=img_root_path,
        vocabulary=vocabulary,
        apply_ignore_list=True,
        blacklist=blacklist,
        max_ds_items=max_ds_items,
    )
    if not vocabulary:
        vocabulary = auto_voc
        vocabulary.save(str(Path(output_model).with_suffix(".voc")))

    dataset = preprocess_dataset(
        dataset,
        only_threshold=False,
        cache_dir=CACHE_DIR_DEFAULT,
        keep=True
    )

    # Fixup shuffler
    # We need to shuffle dataset before split.
    # The purpose is that initially dataset is sorted by
    # writers, and thus if we just dedicate last N of
    # samples we have a risk to remove knowledge about
    # particular handwriting styles totally.
    ds_shuffler = random.Random(DATASET_SHUFFLER_SEED)
    ds_shuffler.shuffle(dataset)

    model = init_model(initial_model_path, vocabulary)
    model.summary()

    total_ds_len = len(dataset)
    train_and_validate_ds_len = int(TRAIN_TEST_RATIO * total_ds_len)
    train_ds_len = train_and_validate_ds_len - TRAIN_VALIDATE_CNT

    train_dataset = dataset[:train_ds_len]
    validate_dataset = dataset[train_ds_len:train_and_validate_ds_len]
    test_dataset = dataset[train_and_validate_ds_len:]

    LOG.info(f"Total dataset size: {total_ds_len}")
    LOG.info(f"    Train set size: {train_ds_len}")
    LOG.info(f"    Validation set size: {train_and_validate_ds_len - train_ds_len}")
    LOG.info(f"    Test set size: {total_ds_len - train_and_validate_ds_len}")

    if validation_list:
        LOG.info(f"Saving validation list to '{validation_list}'...")
        with open(validation_list, "w") as f:
            for gt in tqdm(validate_dataset, desc="Saving validation list"):
                print(f"{gt.img_path}: {gt.str_value}", file=f)

    train_model(
        model, epochs, train_dataset, validate_dataset, vocabulary,
        plot=plot,
    )
    model.save(output_model)

    if plot:
        tf_test_ds = tf_dataset(test_dataset, vocabulary, resize=False)
        tf_plot_predictions(model, tf_test_ds, vocabulary)
