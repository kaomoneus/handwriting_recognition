import argparse
import logging
from pathlib import Path

import numpy as np
from keras.saving.save import load_model
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from dataset_utils import Dataset, tf_dataset, load_dataset
from model_utils import build_model, prediction_model
from plot_utils import plot_samples, plot_predictions
from text_utils import Vocabulary

"""
Defines proportion of training and test data
Training data includes:
* training samples itself - samples we train model on
* validate samples - samples we evaluate progress during training
   passed as additional parameter into 'fit' method.
Test data is used after training is finished.
"""
TRAIN_TEST_RATIO = 0.8

"""
Defines proportion of training and validate samples (both part of samples we use for training)
"""
TRAIN_VALIDATE_RATIO = 0.8

"""
Default number of train epochs
"""
TRAIN_EPOCHS_DEFAULT = 10

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


def handle_train_cmd(args: argparse.Namespace):
    run_train(
        text_path=args.text,
        img_root_path=args.img,
        initial_model_path=args.initial_model,
        output_model=args.output_path,
        epochs=args.epochs
    )


def init_model(model_path: str, vocabulary: Vocabulary) -> keras.Model:
    if not model_path:
        return build_model(vocabulary)

    return load_model(model_path)


def train_model(
    model: keras.Model,
    epochs: int,
    train_ds: Dataset, validate_ds: Dataset,
    vocabulary: Vocabulary
):
    tf_train_ds = tf_dataset(train_ds, vocabulary)
    tf_validation_ds = tf_dataset(validate_ds, vocabulary)

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

            for i in tqdm(range(validation_set_size), desc=f"Evaluating epoch #{epoch}"):
                labels = validation_labels[i]
                predictions = self.prediction_model.predict(
                    validation_images[i],
                    verbose=0
                )
                edit_distances.append(calculate_edit_distance(labels, predictions, vocabulary).numpy())

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


def run_train(text_path, img_root_path, initial_model_path, epochs, output_model):
    dataset, vocabulary = load_dataset(
        str_values_file_path=text_path, img_dir=img_root_path,
        max_word_len=32
    )
    vocabulary.save(str(Path(output_model).with_suffix(".voc")))

    model = init_model(initial_model_path, vocabulary)
    model.summary()

    total_ds_len = len(dataset)
    train_and_validate_ds_len = int(TRAIN_TEST_RATIO * total_ds_len)
    train_ds_len = int(TRAIN_VALIDATE_RATIO * train_and_validate_ds_len)

    train_dataset = dataset[:train_ds_len]
    validate_dataset = dataset[train_ds_len:train_and_validate_ds_len]
    test_dataset = dataset[train_and_validate_ds_len:]

    LOG.info(f"Total dataset size: {total_ds_len}")
    LOG.info(f"    Train set size: {train_ds_len}")
    LOG.info(f"    Validation set size: {train_and_validate_ds_len - train_ds_len}")
    LOG.info(f"    Train set size: {total_ds_len}")

    plot_samples(train_dataset, vocabulary)

    train_model(model, epochs, train_dataset, validate_dataset, vocabulary)
    model.save(output_model)

    plot_predictions(model, test_dataset, vocabulary)
