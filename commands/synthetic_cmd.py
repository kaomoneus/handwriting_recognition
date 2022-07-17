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
from typing import Dict, List

from PIL import ImageFont, ImageDraw, Image
from PIL.ImageFont import FreeTypeFont
from tqdm import tqdm

from errors import Error
from utils.common import Dataset, GroundTruthPathsItem
from utils.dataset_utils import dataset_to_ground_truth, save_ground_truth_json, load_ground_truth_json
from utils.os_utils import list_dir_recursive

SYNTHETIC_FONT_SIZES = [15, 30, 60]

SYNTHETIC_MARGIN = 2

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
    res = render_text(dest, fonts_dir, text_path, max_ds_items)

    gt_path = f"{dest / 'gt.json'}"

    gt = load_ground_truth_json(gt_path) if os.path.exists(gt_path) else {}
    new_gt = dataset_to_ground_truth(res.dataset)
    gt.update(new_gt)
    save_ground_truth_json(gt, gt_path)

    LOG.info(f"Complete.")
    LOG.info(f"    Total lines: {res.num_lines}")
    LOG.info(f"    Total words: {res.num_words}")


@dataclasses.dataclass
class _RenderResult:
    num_lines: int
    num_words: int
    dataset: Dataset


def render_text(dest, fonts_dir, text_path, max_ds_items):

    res: Dataset = []

    # random.seed(datetime.datetime.now().timestamp())
    random.seed(1)
    dest.mkdir(exist_ok=True)
    if not text_path.exists() or text_path.is_dir():
        raise Error(f"File {text_path} doesn't exist or it is directory")
    renderer = Renderer(fonts_dir)
    file_sz = int(os.stat(text_path).st_size / (1024 * 1024))
    progress = tqdm(desc="Rendering", total=file_sz, unit="MB")
    total_words = 0
    line_size = next_line_size()
    cur_line = []
    cur_line_idx = 0
    prev_cur_pos = 0
    num_items_per_render = renderer.num_items_per_render()

    stop_rendering = False

    with open(text_path, "r") as text_file:
        line = text_file.readline()
        while line and not stop_rendering:
            words = line.strip().split()
            for w in words:
                cur_line.append(w)
                if len(cur_line) == line_size:
                    render_res = renderer.render(
                        text=" ".join(cur_line),
                        dest=dest,
                        line_idx=cur_line_idx,
                    )
                    res.extend(render_res)
                    cur_line = []
                    cur_line_idx += 1
                    if max_ds_items and cur_line_idx * num_items_per_render >= max_ds_items:
                        LOG.info(f"Reached limit of {max_ds_items}.")
                        stop_rendering = True
                        break
                    line_size = next_line_size()

            total_words += len(words)

            cur_pos = text_file.tell() / (1024 * 1024)
            upd = int(cur_pos - prev_cur_pos)
            if upd != 0:
                progress.update(upd)
                prev_cur_pos = cur_pos
            line = text_file.readline()

    return _RenderResult(
        num_lines=cur_line_idx,
        num_words=total_words,
        dataset=res,
    )


class Renderer:
    def __init__(self, fonts_dir: Path):
        available_font_paths: List[Path] = list_dir_recursive(
            fonts_dir,
            lambda f: f.suffix in {".ttf", ".otf"}
        )

        self._fonts: Dict[str, FreeTypeFont] = dict()

        for font_path in available_font_paths:
            family = font_path.stem
            index = 0
            while True:
                try:
                    for font_size in SYNTHETIC_FONT_SIZES:
                        font = ImageFont.truetype(str(font_path), font_size, index=index)
                        self._fonts[f"{family}-sz{font_size}-fc{index}"] = font
                    index += 1
                except IOError:
                    break
            if not index:
                LOG.warning(f"Unable to load font file: {font_path}")

    @staticmethod
    def _estimate_size(font: FreeTypeFont, text: str):
        ascent, descent = font.getmetrics()

        mask = font.getmask(text)

        left, top, right, bottom = mask.getbbox()

        return left, ascent - bottom, right, bottom + descent

    def render(self, dest: Path, text: str, line_idx: int):
        res: Dataset = []
        for name, font in self._fonts.items():
            left, top, right, bottom = self._estimate_size(font, text)
            width = right + SYNTHETIC_MARGIN*2
            height = bottom + SYNTHETIC_MARGIN*2

            x = SYNTHETIC_MARGIN - left
            y = SYNTHETIC_MARGIN - top

            img = Image.new("RGB", (width, height), color="white")

            draw_interface = ImageDraw.Draw(img)
            draw_interface.text((x, y), text, font=font, fill="Black")

            img_name = f"{name}-ln{line_idx}"

            img_path = str(dest / f"{img_name}.png")
            gt_txt_path = str(dest / f"{img_name}.gt.txt")
            img.save(img_path)

            with open(gt_txt_path, "w") as gt_txt:
                print(text, file=gt_txt)

            res.append(GroundTruthPathsItem(
                str_value=text,
                img_name=img_name,
                img_path=img_path
            ))

        return res

    def num_items_per_render(self):
        return len(self._fonts)