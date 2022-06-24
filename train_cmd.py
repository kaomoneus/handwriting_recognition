import argparse
import logging
import pathlib
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.saving.save import load_model
from tensorflow import keras
from tqdm import tqdm

from config import CACHE_DIR_DEFAULT, TRAIN_EPOCHS_DEFAULT, TRAIN_TEST_RATIO, TRAIN_VALIDATE_CNT, DATASET_SHUFFLER_SEED, \
    TENSORBOARD_LOGS_DEFAULT, BATCH_SIZE_DEFAULT
from dataset_utils import Dataset, tf_dataset, load_dataset, preprocess_dataset, load_marked, parse_dataset_args, \
    add_dataset_args, save_marked, MarkedState
from model_utils import build_model, EditDistanceCallback
from plot_utils import tf_plot_predictions, plot_interactive
from text_utils import Vocabulary, add_voc_args, parse_voc_args

LOG = logging.getLogger(__name__)


# TODO:
#   1. rename all command related calls into:
#      register, handle, ...
#      without mentioning command name
#   2. Use aggregation in main, to get all modules from 'commands' submudule.
def register_train_args(train_cmd: argparse.ArgumentParser):
    train_cmd.add_argument(
        "-i", dest="initial_model",
        help="Initial model path. If provided then saved model will be loaded."
    )
    train_cmd.add_argument(
        "-o", dest="output_path",
        help="Output model path. Model will be trained and saved at this path."
    )
    train_cmd.add_argument(
        "-e", dest="epochs",
        default=TRAIN_EPOCHS_DEFAULT,
        type=int,
        help="Number of train epochs."
    )
    train_cmd.add_argument(
        "-b", dest="batch_size",
        default=BATCH_SIZE_DEFAULT,
        type=int,
        help="Number of samples per batch."
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

    add_voc_args(train_cmd)
    add_dataset_args(train_cmd)


def handle_train_cmd(args: argparse.Namespace):
    vocabulary = parse_voc_args(args)
    dataset, _ = parse_dataset_args(args, vocabulary)
    run_train(
        dataset=dataset,
        initial_model_path=args.initial_model,
        output_model=args.output_path,
        epochs=args.epochs,
        validation_list=args.validation_list,
        plot=args.plot,
        vocabulary=parse_voc_args(args),
        batch_size=args.batch_size,
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
    batch_size: int,
):
    tf_train_ds = tf_dataset(train_ds, vocabulary, resize=False, batch_size=batch_size)
    tf_validation_ds = tf_dataset(validate_ds, vocabulary, resize=False, batch_size=batch_size)

    validation_images = []
    validation_labels = []

    for batch in tf_validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    edit_distance_callback = EditDistanceCallback(
        model,
        validation_images, validation_labels,
        vocabulary
    )

    log_dir = pathlib.Path(TENSORBOARD_LOGS_DEFAULT) / datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=str(log_dir), histogram_freq=1)
    file_writer = tf.summary.create_file_writer(str(log_dir / "metrics"))
    file_writer.set_as_default()

    # Train the model.
    history = model.fit(
        tf_train_ds,
        validation_data=tf_validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback, tensorboard_callback],
    )

    return history


def run_train(
    dataset: Dataset,
    initial_model_path,
    epochs,
    output_model,
    validation_list,
    plot,
    vocabulary: Optional[Vocabulary],
    batch_size: int,
):
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
    LOG.info(f"    Samples per batch: {batch_size}")

    if epochs and not output_model:
        LOG.warning(
            f"You requested training, but didn't specify 'output_path'"
            f" trained model WILL NOT BE SAVED."
        )

    if validation_list:
        LOG.info(f"Saving validation list to '{validation_list}'...")
        save_marked(
            state_path=validation_list,
            state=MarkedState(marked=[gt.img_name for gt in validate_dataset])
        )

    if plot:
        plot_interactive(train_dataset, 8, 8)

    if epochs:
        train_model(
            model, epochs, train_dataset, validate_dataset, vocabulary,
            batch_size,
        )
        if output_model:
            model.save(output_model)

    if plot:
        tf_test_ds = tf_dataset(test_dataset, vocabulary, resize=False, batch_size=batch_size)
        tf_plot_predictions(model, tf_test_ds, vocabulary)
