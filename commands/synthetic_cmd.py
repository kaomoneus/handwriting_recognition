"""
Exports dataset, currently only exports to tesseract format
"""

import argparse
import dataclasses
import datetime
import hashlib
import logging
import os
import random
from pathlib import Path
from typing import Dict, List, Set

from PIL import ImageFont, ImageDraw, Image
from PIL.ImageFont import FreeTypeFont
from tqdm import tqdm

from errors import Error
from utils.common import Dataset, GroundTruthPathsItem
from utils.dataset_utils import dataset_to_ground_truth, save_ground_truth_json, load_ground_truth_json
from utils.os_utils import list_dir_recursive
from utils.text_utils import add_voc_args, parse_voc_args

SYNTHETIC_FONT_SIZES = [15, 30, 60]
SYNTHETIC_MARGIN = 2
KNOWN_SPACE_LIKE_SYMBOLS = {' '}

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
    add_voc_args(parser)


def next_line_size():
    return random.randint(1, 5)


def handle(args: argparse.Namespace):
    dest = Path(args.dest)
    text_path = Path(args.text)
    fonts_dir = Path(args.fonts_dir)
    max_ds_items = args.max_ds_items
    vocabulary = parse_voc_args(args)
    res = render_text(dest, fonts_dir, text_path, max_ds_items, vocabulary)

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


def render_text(dest, fonts_dir, text_path, max_ds_items, vocabulary):

    res: Dataset = []

    # random.seed(datetime.datetime.now().timestamp())
    random.seed(1)
    dest.mkdir(exist_ok=True)
    if not text_path.exists() or text_path.is_dir():
        raise Error(f"File {text_path} doesn't exist or it is directory")
    renderer = Renderer(fonts_dir)
    scale = 1024
    file_sz = int(os.stat(text_path).st_size / scale)
    progress = tqdm(desc="Rendering", total=file_sz, unit="Kb")
    total_words = 0
    line_size = next_line_size()
    cur_line = []
    cur_line_idx = 0
    prev_upd_pos = 0
    num_items_per_render = renderer.num_items_per_render()

    dest_hash = str(hashlib.md5(str(dest).encode("utf-8")).hexdigest())

    stop_rendering = False
    cur_pos = 0

    with open(text_path, "r") as text_file:
        line = text_file.readline()
        while line and not stop_rendering:
            words = [
                w
                for w in line.strip().split()
                if vocabulary is None or not set(w).difference(vocabulary.characters)
            ]
            for w in words:
                cur_line.append(w)
                total_words += 1
                if len(cur_line) == line_size:
                    text = " ".join(cur_line)
                    unique_prefix = f"{dest_hash[:5]}-{total_words-len(cur_line):04x}-{len(cur_line):02x}"
                    render_res = renderer.render(
                        text=text,
                        dest=dest,
                        unique_prefix=unique_prefix,
                    )
                    if not render_res:
                        LOG.warning(f"Unable to render text: '{text}'")
                    else:
                        res.extend(render_res)
                    cur_line = []
                    cur_line_idx += 1
                    if max_ds_items and len(res) >= max_ds_items:
                        LOG.info(f"Reached limit of {max_ds_items}.")
                        stop_rendering = True
                        break
                    line_size = next_line_size()

            cur_pos = text_file.tell()
            upd = int((cur_pos - prev_upd_pos)/scale)
            if upd != 0:
                progress.update(upd)
                prev_upd_pos = cur_pos
            line = text_file.readline()

    return _RenderResult(
        num_lines=cur_line_idx,
        num_words=total_words,
        dataset=res,
    )


@dataclasses.dataclass()
class FontInfo:
    pillow_font: FreeTypeFont
    supported_characters: Set[str] = dataclasses.field(default_factory=set)
    unsupported_characters: Set[str] = dataclasses.field(default_factory=set)

    def check_supported(self, text: str):
        text = {symbol for symbol in text if symbol not in KNOWN_SPACE_LIKE_SYMBOLS}

        if text.intersection(self.unsupported_characters):
            return False

        if len(text.difference(self.supported_characters)) == 0:
            return True

        all_supported = True

        for symbol in text:
            mask = self.pillow_font.getmask(symbol)
            bbox = mask.getbbox()
            if bbox is None:
                self.unsupported_characters.add(symbol)
                all_supported = False
                continue
            self.supported_characters.add(symbol)

        return all_supported



class Renderer:
    def __init__(self, fonts_dir: Path):
        available_font_paths: List[Path] = list_dir_recursive(
            fonts_dir,
            lambda f: f.suffix in {".ttf", ".otf"}
        )

        self._fonts: Dict[str, FontInfo] = dict()

        for font_path in available_font_paths:
            family = font_path.stem
            index = 0
            while True:
                try:
                    for font_size in SYNTHETIC_FONT_SIZES:
                        font = ImageFont.truetype(str(font_path), font_size, index=index)
                        self._fonts[f"{family}-sz{font_size}-fc{index}"] = FontInfo(
                            pillow_font=font,
                        )
                    index += 1
                except IOError:
                    break
            if not index:
                LOG.warning(f"Unable to load font file: {font_path}")

    @staticmethod
    def _estimate_size(font: FontInfo, text: str):
        if not font.check_supported(text):
            return None

        pillow_font = font.pillow_font
        ascent, descent = pillow_font.getmetrics()

        mask = pillow_font.getmask(text)

        bbox = mask.getbbox()
        if bbox is None:
            return None

        left, top, right, bottom = bbox
        return left, ascent - bottom, right, bottom + descent

    def render(self, dest: Path, text: str, unique_prefix: str):
        res: Dataset = []
        for name, font in self._fonts.items():
            pillow_font = font.pillow_font
            img_name = f"{unique_prefix}-{name}"
            est = self._estimate_size(font, text)
            if est is None:
                LOG.debug(f"Skipped '{img_name}' for text '{text}'")
                continue
            left, top, right, bottom = est
            width = right + SYNTHETIC_MARGIN*2
            height = bottom + SYNTHETIC_MARGIN*2

            x = SYNTHETIC_MARGIN - left
            y = SYNTHETIC_MARGIN - top

            img = Image.new("RGB", (width, height), color="white")

            draw_interface = ImageDraw.Draw(img)
            draw_interface.text((x, y), text, font=pillow_font, fill="Black")

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