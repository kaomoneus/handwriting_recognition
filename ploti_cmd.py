import argparse
import dataclasses
import json
import logging
import pathlib
from typing import List

from config import PLOTI_ROWS, PLOTI_COLS, MARKED_PATH_DEFAULT, CACHE_DIR_DEFAULT
from dataset_utils import load_dataset, load_marked, save_marked, MarkedState, preprocess_dataset, add_dataset_args, \
    parse_dataset_args, Dataset
from plot_utils import plot_interactive
from text_utils import add_voc_args, parse_voc_args, Vocabulary


LOG = logging.getLogger(__name__)


def register_ploti_args(ploti_cmd: argparse.ArgumentParser):
    ploti_cmd.add_argument("-state", help="File with current state", default=MARKED_PATH_DEFAULT)

    add_voc_args(ploti_cmd)
    add_dataset_args(ploti_cmd)


def handle_ploti_cmd(args: argparse.Namespace):
    vocabulary = parse_voc_args(args)
    dataset, _ = parse_dataset_args(args, vocabulary)

    run_ploti(
        dataset=dataset,
        vocabulary=vocabulary,
        state_path=args.state
    )


def run_ploti(dataset: Dataset, vocabulary: Vocabulary, state_path: str):
    # Resize, threshold and pad samples per network input configuration.
    dataset = preprocess_dataset(
        dataset,
        only_threshold=True,
        cache_dir=CACHE_DIR_DEFAULT,
    )

    current_page = 0
    samples_per_page = PLOTI_ROWS * PLOTI_COLS
    marked = set()
    ignore = set(vocabulary.ignore) if vocabulary.ignore else None

    if pathlib.Path(state_path).exists():
        state = load_marked(state_path)
        marked = set(state.marked)
        current_item_idx_list = [i for i, x in enumerate(dataset) if x.img_name == state.start_item]
        if len(current_item_idx_list):
            current_page = current_item_idx_list[0] // samples_per_page
            if current_page != state.current_page:
                LOG.warning("Calculated current page differs from saved one. Using the former.")
        else:
            LOG.warning("Saved item name not found in dataset. Using saved page number.")
            current_page = state.current_page

    def on_save(current_page: int, start_item: str):
        save_marked(state_path, MarkedState(
            marked=list(sorted(marked)),
            current_page=current_page,
            start_item=start_item
        ))
        LOG.info(f"Page #{current_page}, state saved at '{state_path}'")

    plot_interactive(
        dataset,
        PLOTI_ROWS, PLOTI_COLS,
        marked=marked,
        ignored=ignore,
        start_page=current_page,
        on_page_changed=on_save,
        on_save=on_save
    )
