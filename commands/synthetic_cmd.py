"""
Exports dataset, currently only exports to tesseract format
"""

import argparse
import dataclasses
import datetime
import logging
import os
import random
from pathlib import Path

from tqdm import tqdm

from config import MAX_WORD_LEN_DEFAULT
from errors import Error
from utils.dataset_utils import add_blacklist_args, parse_blacklist_args, make_lines_dataset, WordsFilter, \
    save_ground_truth_json, \
    dataset_to_ground_truth, GTFormat
from utils.text_utils import add_voc_args, parse_voc_args

LOG = logging.getLogger()


def register(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-dest",
        required=True,
        help="Destination directory where rendered lines will be stored"
    )
    parser.add_argument(
        "-text",
        required=True,
        help="Path to file with text."
    )
    parser.add_argument(
        "-fonts-dir",
        required=True,
        help="Directory with forms XML files given in IAM format."
    )
    parser.add_argument(
        "-max-ds-items",
        type=int,
        help="Maximum amount of loaded datasource items"
    )


def next_line_size():
    return random.randint(1, 5)


def handle(args: argparse.Namespace):
    dest = Path(args.dest)
    text_path = Path(args.text)
    fonts_dir = Path(args.fonts_dir)
    max_ds_items = args.max_ds_items
    random.seed(datetime.datetime.now().timestamp())

    class Renderer:
        def __init__(self, fonts_dir: Path):
            pass

        def render(self, text: str, line_idx: int):
            pass

    dest.mkdir(exist_ok=True)

    if not text_path.exists() or text_path.is_dir():
        raise Error(f"File {text_path} doesn't exist or it is directory")

    renderer = Renderer(fonts_dir)

    file_sz = int(os.stat(text_path).st_size / (1024*1024))
    progress = tqdm(desc="Rendering", total=file_sz, unit="MB")

    line_size = next_line_size()
    cur_line = []
    cur_line_idx = 0
    prev_cur_pos = 0
    with open(text_path, "r") as text_file:
        line = text_file.readline()
        while line:
            words = line.split()
            for w in words:
                cur_line.append(w)
                if len(cur_line) == line_size:
                    renderer.render(
                        text=" ".join(cur_line),
                        line_idx=cur_line_idx
                    )
                    cur_line = []
                    cur_line_idx += 1
                    line_size = next_line_size()

            cur_pos = text_file.tell() / (1024*1024)
            upd = int(cur_pos - prev_cur_pos)
            if upd != 0:
                progress.update(upd)
                prev_cur_pos = cur_pos
            line = text_file.readline()
