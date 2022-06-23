import argparse
import dataclasses
import json
import logging
import pathlib
from typing import List

from config import PLOTI_ROWS, PLOTI_COLS
from dataset_utils import load_dataset
from plot_utils import plot_interactive
from text_utils import add_voc_args, parse_voc_args, Vocabulary


LOG = logging.getLogger(__name__)


def register_ploti_args(ploti_cmd: argparse.ArgumentParser):
    ploti_cmd.add_argument("-img", help="Root directory with images", required=True)
    ploti_cmd.add_argument("-text", help="File with text ground truth", required=True)
    ploti_cmd.add_argument("-state", help="File with current state", default=".plotistate.json")

    add_voc_args(ploti_cmd)


def handle_ploti_cmd(args: argparse.Namespace):
    run_ploti(
        img_path=args.img,
        text_path=args.text,
        vocabulary=parse_voc_args(args),
        state_path=args.state
    )


def run_ploti(img_path: str, text_path: str, vocabulary: Vocabulary, state_path: str):

    dataset, _ = load_dataset(
        str_values_file_path=text_path,
        img_dir=img_path,
        vocabulary=vocabulary,
    )

    current_page = 0
    samples_per_page = PLOTI_ROWS * PLOTI_COLS
    marked = set()
    ignore = set(vocabulary.ignore) if vocabulary.ignore else None

    @dataclasses.dataclass
    class State:
        marked: List[str] = dataclasses.field(default_factory=list)
        current_page: int = 0
        start_item: str = dataset[0].img_name

    if pathlib.Path(state_path).exists():
        with open(state_path, "r") as ff:
            vv = json.load(ff)
            state = State(**vv)
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
        v = dataclasses.asdict(State(
            marked=list(sorted(marked)),
            current_page=current_page,
            start_item=start_item
        ))
        with open(state_path, "w") as f:
            json.dump(v, f, indent=4)
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
