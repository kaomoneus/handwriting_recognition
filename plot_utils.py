import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset_utils import Dataset
from model_utils import prediction_model
from text_utils import Vocabulary, PADDING_TOKEN

LOG = logging.getLogger()


def onclick_handler_default(event):
    click_type = 'double' if event.dblclick else 'single'
    text: plt.Text = event.inaxes.title
    LOG.info(
        f"{click_type} Click: button {event.button}, x={event.x}, y={event.y}"
    )
    LOG.info(
        f"     subplot '{text.get_text()}', at: {event.xdata}, {event.ydata}"
    )


def make_subplots(onclick=onclick_handler_default):
    figure, ax = plt.subplots(8, 8, figsize=(15, 8))
    figure: plt.Figure = figure
    canvas: plt.FigureCanvasBase = figure.canvas

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            axx: plt.Axes = ax[i, j]
            axx.set_frame_on(True)
            axx.tick_params(
                axis="both",
                which='both',
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labelleft=False
            )

    if onclick:
        canvas.mpl_connect('button_press_event', onclick_handler_default)
    return ax


def set_subplot_img(ax: np.ndarray, row: int, col: int, img: np.ndarray, title: str):
    axx = ax[row, col]
    axx.imshow(img, cmap="gray")
    text = axx.set_title(title, fontdict=dict(fontsize=6))
    return text


def tf_plot_samples(tf_ds, vocabulary: Vocabulary):
    """
        ## Visualize a few samples
        """

    for data in tf_ds.take(1):
        images, labels = data["image"], data["label"]

        ax = make_subplots()

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

            set_subplot_img(ax, i // 4, i % 4, img, label)

    plt.show()


# A utility function to decode the output of the network.
def _decode_batch_predictions(pred, vocabulary: Vocabulary):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search.
    results = tf.keras.backend.ctc_decode(
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


def tf_plot_predictions(model: tf.keras.Model, tf_ds, vocabulary: Vocabulary):

    #  Let's check results on some test samples.
    predictor = prediction_model(model)
    for batch in tf_ds.take(1):
        batch_images = batch["image"]
        ax = make_subplots()

        preds = predictor.predict(batch_images)
        pred_texts = _decode_batch_predictions(preds, vocabulary)

        for i in range(min(16, len(batch_images))):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"

            set_subplot_img(ax, i // 4, i % 4, img, title)

    plt.show()


def plot_dataset(dataset: Dataset):
    """
    Plots up to 16 dataset samples
    :param dataset:
    :return:
    """

    ax = make_subplots()

    for i, gt in enumerate(dataset[:min(len(dataset), 16)]):
        title = f"{gt.img_name}: '{gt.str_value}'" if gt.str_value else gt.img_name
        img = cv2.imread(gt.img_path, flags=cv2.IMREAD_GRAYSCALE)
        set_subplot_img(ax, i // 4, i % 4, img, title)

    plt.show()
