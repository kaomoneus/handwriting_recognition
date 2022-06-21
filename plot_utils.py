import dataclasses
import logging
from typing import Union, List, Set, Callable

import cv2
import matplotlib.pyplot as plt
import matplotlib.spines as spines
import numpy as np
import tensorflow
import tensorflow as tf
from matplotlib.backend_bases import MouseEvent, KeyEvent

from dataset_utils import Dataset, ROI
from model_utils import prediction_model
from text_utils import Vocabulary, PADDING_TOKEN

LOG = logging.getLogger()


@dataclasses.dataclass
class PlotTask:
    title: str
    image: Union[np.ndarray, tensorflow.Tensor]
    roi: ROI = None
    marked: bool = False


def onclick_handler_default(event):
    click_type = 'double' if event.dblclick else 'single'
    LOG.info(
        f"{click_type} Click: button {event.button}, x={event.x}, y={event.y}"
    )
    if event.inaxes is not None:
        text: plt.Text = event.inaxes.title
        LOG.info(
            f"     subplot '{text.get_text()}', at: {event.xdata}, {event.ydata}"
        )


def make_subplots(
    onclick=onclick_handler_default,
    onkeypress=None,
    rows=4, cols=4
):
    figure, axes_grid = plt.subplots(rows, cols, figsize=(15, 8))
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

    if onclick:
        canvas.mpl_connect('button_press_event', onclick)
    if onkeypress:
        canvas.mpl_connect('key_press_event', onkeypress)

    return axes_grid


def set_subplot_img(ax: np.ndarray, row: int, col: int, img: np.ndarray, title: str, marked):
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


def plot_interactive(
    dataset: Dataset,
    rows: int, cols: int,
    marked: Set[str] = None,
    on_mark: Callable[[str, bool, plt.Axes], None] = None,
    on_page_changed: Callable[[int, str], None] = None,
    start_page: int = 0
):
    """
    Plots dataset samples interactively
    :param dataset: dataset which samples to be plotted
    :param rows: amount of rows with plotted samples
    :param cols: amount of cols with plotted samples
    :param marked: collection of marked items, mutable
    :param on_mark: callback, called when user marked or unmarked som item
    :param on_page_changed: called when user changed page, passes two arguments: page number and item nameß
    :param start_page: page
    :return:
    """

    current_page = start_page
    samples_per_page = rows * cols
    max_page = (len(dataset) + 1) // samples_per_page

    subplots = None
    if marked is None:
        marked = set()

    def plot_current_page():
        sample_start = samples_per_page * current_page
        sample_end = sample_start + samples_per_page
        tasks = [
            PlotTask(
                image=cv2.imread(gt.img_path, flags=cv2.IMREAD_GRAYSCALE),
                title=f"{gt.img_name}: '{gt.str_value}'" if gt.str_value else gt.img_name,
                marked=(gt.img_name in marked)
            )
            for gt in dataset[sample_start:sample_end]
        ]

        plot_tasks(subplots, tasks)
        plt.draw()

    def on_mark_default(name: str, marked: bool, ax: plt.Axes):
        LOG.info(f"Item '{name}' was {'marked' if marked else 'unmarked'}")
        for spine in ax.spines.values():
            spine: spines.Spine
            if marked:
                spine.set_edgecolor('red')
                spine.set_linewidth(2.5)
            else:
                spine.set_edgecolor('black')
                spine.set_linewidth(0.5)
        plt.draw()

    def on_page_change_default(page: int):
        LOG.info(f"Page changed to #{page}")

    if on_mark is None:
        on_mark = on_mark_default

    if on_page_changed is None:
        on_page_changed = on_page_change_default

    def on_mouse_click(event: MouseEvent):
        if event.dblclick:
            return

        if event.inaxes is None:
            return

        text: plt.Text = event.inaxes.title
        name = text.get_text().split(":")[0]

        if name in marked:
            marked.remove(name)
            on_mark(name, False, event.inaxes)
        else:
            marked.add(name)
            on_mark(name, True, event.inaxes)

    def on_key_pressed(event: KeyEvent):
        nonlocal current_page
        key = event.key
        LOG.info(f"Key pressed: {key}")
        if key in {"right", "]"}:
            current_page = min(current_page + 1, max_page)
        if key in {"left", "["}:
            current_page = max(current_page - 1, 0)
        plot_current_page()
        plt.show()

    subplots = make_subplots(
        onclick=on_mouse_click,
        onkeypress=on_key_pressed,
        rows=rows,
        cols=cols,
    )
    plot_current_page()
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
        set_subplot_img(subplots, row, col, img, title, pt.marked)
