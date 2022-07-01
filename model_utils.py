
"""
You will notice that the content of original image is kept as faithful as possible and has
been padded accordingly.
"""
import gc
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow.python.data
from keras.backend import clear_session
from keras.saving.save import load_model
from tensorflow import keras
from tqdm import tqdm

from dataset_utils import Dataset, tf_dataset
from config import IMAGE_WIDTH, IMAGE_HEIGHT
from text_utils import Vocabulary, PADDING_TOKEN

"""
## Model

Our model will use the CTC loss as an endpoint layer. For a detailed understanding of the
CTC loss, refer to [this post](https://distill.pub/2017/ctc/).
"""


LOG = logging.getLogger(__name__)


class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred


def build_model(vocabulary: Vocabulary):

    # Inputs to the model
    input_img = keras.Input(shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 1), name="image")
    labels = keras.layers.Input(name="label", shape=(None,))

    # 0th conv block.
    x = keras.layers.Conv2D(
        16,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv0",
    )(input_img)
    x = keras.layers.MaxPooling2D((2, 2), name="pool0")(x)

    # First conv block.
    x = keras.layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block.
    x = keras.layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = keras.layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model.
    new_shape = ((IMAGE_WIDTH // 8), (IMAGE_HEIGHT // 8) * 64)
    x = keras.layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = keras.layers.Dense(64, activation="relu", name="dense1")(x)
    x = keras.layers.Dropout(0.2)(x)

    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(128, return_sequences=True, dropout=0.25)
    )(x)
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(64, return_sequences=True, dropout=0.25)
    )(x)

    # +2 is to account for the two special tokens introduced by the CTC loss.
    # The recommendation comes here: https://git.io/J0eXP.
    x = keras.layers.Dense(
        len(vocabulary.char_to_num.get_vocabulary()) + 2, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step.
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model.
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="handwriting_recognizer"
    )
    # Optimizer.
    opt = keras.optimizers.Adam()
    # Compile the model and return.
    model.compile(optimizer=opt, run_eagerly=True)
    return model


def prediction_model(model: keras.Model):
    return keras.models.Model(
        model.get_layer(name="image").input, model.get_layer(name="dense2").output
    )


def init_model(model_path: str, vocabulary: Vocabulary, image_width: int, image_height: int) -> keras.Model:
    if not model_path:
        return build_model(image_width, image_height, vocabulary)

    return load_model(model_path)


def train_model(
    model: keras.Model,
    epochs: int,
    train_ds: Dataset, validate_ds: Dataset,
    batch_size: int,
    vocabulary: Vocabulary
):
    tf_train_ds = tf_dataset(train_ds, vocabulary, batch_size, resize=False)
    tf_validation_ds = tf_dataset(validate_ds, vocabulary, batch_size, resize=False)

    validation_images = []
    validation_labels = []

    for batch in tf_validation_ds:
        validation_images.append(batch["image"])
        validation_labels.append(batch["label"])

    validation_set_size = len(validation_images)

    edit_distance_callback = EditDistanceCallback(model, validation_images, validation_labels, vocabulary)
    clear_callback = ClearMemory()

    # Train the model.
    history = model.fit(
        tf_train_ds,
        validation_data=tf_validation_ds,
        epochs=epochs,
        callbacks=[edit_distance_callback, clear_callback],
    )

    return history


def calculate_edit_distance(labels, predictions, vocabulary: Vocabulary):
    # Get a single batch and convert its labels to sparse tensors.
    labels = tf.where(
        tf.equal(labels, PADDING_TOKEN),
        x=tf.constant(-1, labels.dtype, labels.shape),
        y=labels
    )
    sparse_labels: tf.SparseTensor = tf.cast(tf.sparse.from_dense(labels), dtype=tf.int64)

    # Make predictions and convert them to sparse tensors.
    input_len = np.ones(predictions.shape[0]) * predictions.shape[1]
    predictions_decoded: tf.Tensor = tf.keras.backend.ctc_decode(
        predictions, input_length=input_len, greedy=True
    )[0][0][:, :vocabulary.max_len]
    sparse_predictions = tf.cast(
        tf.sparse.from_dense(predictions_decoded), dtype=tf.int64
    )

    # Compute individual edit distances and average them out.
    edit_distances = tf.edit_distance(
        sparse_predictions, sparse_labels, normalize=False
    )
    return tf.reduce_mean(edit_distances)


class EditDistanceCallback(keras.callbacks.Callback):
    def __init__(
        self,
        model,
        validation_images,
        validation_labels,
        vocabulary: Vocabulary,
    ):
        super().__init__()
        self.prediction_model = prediction_model(model)
        self.validation_images = validation_images
        self.validation_labels = validation_labels
        self.validation_set_size = len(validation_labels)
        self.vocabulary = vocabulary

    def on_epoch_end(self, epoch, logs=None):
        edit_distances = []

        for i in tqdm(range(self.validation_set_size), desc=f"Evaluating epoch #{epoch}"):
            labels = self.validation_labels[i]
            images = self.validation_images[i]

            predictions = self.prediction_model.predict(images, verbose=0)
            edit_distances.append(calculate_edit_distance(labels, predictions, self.vocabulary).numpy())

            clear_session()

        med = np.mean(edit_distances)

        LOG.info(f"Mean edit distance for epoch {epoch + 1}: {med:.4f}")
        tf.summary.scalar("MED", data=med, step=epoch, description="Mean edit distance")


class ClearMemory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        clear_session()
