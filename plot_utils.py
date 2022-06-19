import dataclasses
import logging
from typing import Union, List

import cv2
import matplotlib.pyplot as plt
import matplotlib.spines as spines
import numpy as np
import tensorflow
import tensorflow as tf

from dataset_utils import Dataset, ROI
from model_utils import prediction_model
from text_utils import Vocabulary, PADDING_TOKEN

LOG = logging.getLogger()


@dataclasses.dataclass
class PlotTask:
    title: str
    image: Union[np.ndarray, tensorflow.Tensor]
    roi: ROI = None


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
    figure, axes_grid = plt.subplots(8, 8, figsize=(15, 8))
    figure: plt.Figure = figure
    canvas: plt.FigureCanvasBase = figure.canvas

    for i in range(axes_grid.shape[0]):
        for j in range(axes_grid.shape[1]):
            axx: plt.Axes = axes_grid[i, j]
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
            for spine in axx.spines.values():
                spine: spines.Spine
                spine.set_edgecolor('green')
                spine.set_linewidth(2.5)

    if onclick:
        canvas.mpl_connect('button_press_event', onclick_handler_default)
    return axes_grid


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
        tasks: List[PlotTask] = []

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

            tasks.append(PlotTask(
                title=label, image=img
            ))

        plot_tasks(ax, tasks)
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

    ax = make_subplots()

    tasks: List[PlotTask] = []

    for batch in tf_ds.take(1):
        batch_images = batch["image"]

        preds = predictor.predict(batch_images)
        pred_texts = _decode_batch_predictions(preds, vocabulary)

        for i in range(min(16, len(batch_images))):
            img = batch_images[i]
            img = tf.image.flip_left_right(img)
            img = tf.transpose(img, perm=[1, 0, 2])
            img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
            img = img[:, :, 0]

            title = f"Prediction: {pred_texts[i]}"

            tasks.append(PlotTask(
                title=title,
                image=img
            ))

    plot_tasks(ax, tasks)
    plt.show()


def plot_dataset(dataset: Dataset):
    """
    Plots up to 16 dataset samples
    :param dataset:
    :return:
    """

    axes_grid = make_subplots()

    tasks = [
        PlotTask(
            image=cv2.imread(gt.img_path, flags=cv2.IMREAD_GRAYSCALE),
            title=f"{gt.img_name}: '{gt.str_value}'" if gt.str_value else gt.img_name,
        )
        for gt in dataset
    ]

    plot_tasks(axes_grid, tasks)

    plt.show()


def plot_tasks(subplots, plot_tasks: List[PlotTask]):
    """
    Plots samples as described in plot_tasks
    :param subplots: destination with subplots
    :param plot_tasks: tasks to be plotted
    """
    rows, cols = tuple(subplots.shape)

    max_images = rows * cols
    tasks = plot_tasks if len(plot_tasks) <= max_images else plot_tasks[:max_images]

    for i, pt in enumerate(tasks):
        title = pt.title
        img = pt.image
        row = i // cols
        col = i % cols
        set_subplot_img(subplots, row, col, img, title)
