import argparse
import logging
import os

import keras
import numpy as np
from keras.saving.save import load_model
from tqdm import tqdm

from dataset_utils import Dataset, load_dataset, tf_dataset, extract_images_and_labels
from model_utils import build_model, prediction_model
import tensorflow as tf

from text_utils import Vocabulary, PADDING_TOKEN
import matplotlib.pyplot as plt

"""
Defines internal format of rendered strings.
Note, if string is quite short for such aspect ration
it is supposed to pad such rendered string image.
"""
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 32

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

"""
Training batch size
"""
BATCH_SIZE = 64

LOG = logging.getLogger(__name__)


class Error(Exception):
    def __init__(self, message):
        self.message = message


def init_model(model_path: str, vocabulary: Vocabulary) -> keras.Model:
    if not model_path:
        return build_model(IMAGE_WIDTH, IMAGE_HEIGHT, vocabulary)

    return load_model(model_path)


def train_model(
    model: keras.Model,
    epochs: int,
    train_ds: Dataset, validate_ds: Dataset,
    vocabulary: Vocabulary
):
    tf_train_ds = tf_dataset(train_ds, vocabulary, BATCH_SIZE)
    tf_validation_ds = tf_dataset(validate_ds, vocabulary, BATCH_SIZE)

    validation_images = []
    validation_labels = []
    validation_set_size = len(validation_images)

    for batch in tf_validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

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


# A utility function to decode the output of the network.
def _decode_batch_predictions(pred, vocabulary: Vocabulary):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = keras.backend.ctc_decode(
        pred,
        input_length=input_len, greedy=True
    )[0][0][:, :vocabulary.max_len]
    # Iterate over the results and get back the text.
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(vocabulary.num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def plot_samples(ds: Dataset, vocabulary: Vocabulary):
    """
        ## Visualize a few samples
        """

    tf_ds = tf_dataset(ds, vocabulary, BATCH_SIZE)

    for data in tf_ds.take(1):
        images, labels = data["image"], data["label"]

        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        for i in range(16):
            img = images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            # Gather indices where label!= padding_token.
            label = labels[i]
            indices = tf.gather(label, tf.where(tf.math.not_equal(label, PADDING_TOKEN)))
            # Convert to string.
            label = tf.strings.reduce_join(vocabulary.num_to_char(indices))
            label = label.numpy().decode("utf-8")

            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(label)
            ax[i // 4, i % 4].axis("off")

    plt.show()


def plot_predictions(model: keras.Model, dataset: Dataset, vocabulary: Vocabulary):
    tf_ds = tf_dataset(dataset, vocabulary, BATCH_SIZE)

    #  Let's check results on some test samples.
    predictor = prediction_model(model)
    for batch in tf_ds.take(1):
        batch_images = batch["image"]
        _, ax = plt.subplots(4, 4, figsize=(15, 8))

        preds = predictor.predict(batch_images)
        pred_texts = _decode_batch_predictions(preds, vocabulary)

        for i in range(16):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")

    plt.show()
    pass


def run_train(text_path, img_root_path, initial_model_path, epochs, output_model):
    dataset, vocabulary = load_dataset(
        str_values_file_path=text_path, img_dir=img_root_path,
        image_width=IMAGE_WIDTH, image_height=IMAGE_HEIGHT,
        max_word_len=32
    )

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


def main():
    try:
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="cmd")
        eval_cmd = subparsers.add_parser("eval", help="Runs neural network evaluation")
        train_cmd = subparsers.add_parser("train", help="Runs neural network training and then evaluation")
        recognize_cmd = subparsers.add_parser("recognize", help="Runs text recognition")

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

        args = parser.parse_args()

        if args.cmd == "train":
            run_train(
                text_path=args.text,
                img_root_path=args.img,
                initial_model_path=args.initial_model,
                output_model=args.output_path,
                epochs=args.epochs
            )

        return 0

    except Error as e:
        LOG.error(f"Error: {e.message}")
        return 1
