"""
Exports dataset, currently only exports to tesseract format
"""

import argparse
import logging
from pathlib import Path

from config import MAX_WORD_LEN_DEFAULT
from utils.dataset_utils import add_blacklist_args, parse_blacklist_args, make_lines_dataset, WordsFilter, \
    save_ground_truth_json, \
    dataset_to_ground_truth, GTFormat
from utils.text_utils import add_voc_args, parse_voc_args

LOG = logging.getLogger()


def register(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-lines-img-root",
        help="Destination directory where rendered lines will be stored"
    )
    parser.add_argument(
        "-forms-xml-dir",
        help="Directory with forms XML files given in IAM format."
    )
    parser.add_argument(
        "-image-dir",
        help="Directory with forms XML files given in IAM format."
    )
    parser.add_argument(
        "-max-ds-items",
        type=int,
        help="Maximum amount of loaded datasource items"
    )
    add_blacklist_args(parser)
    add_voc_args(parser)


def handle(args: argparse.Namespace):
    vocabulary = parse_voc_args(args)
    blacklist, whitelist = parse_blacklist_args(args)

    filter = WordsFilter(
        blacklist=blacklist, whitelist=whitelist,
        ignore_list=vocabulary.ignore if vocabulary else None,
        allowed_characters=vocabulary.characters if vocabulary else None,
        max_word_len=vocabulary.max_len if vocabulary else MAX_WORD_LEN_DEFAULT,
        max_ds_items=args.max_ds_items,
    )

    lines_root = Path(args.lines_img_root)
    lines_root.mkdir(exist_ok=True)

    ds = make_lines_dataset(
        lines_root=lines_root,
        forms_xml_root=Path(args.forms_xml_dir),
        forms_img_root=Path(args.image_dir),
        filter=filter,
    )
    gt = dataset_to_ground_truth(ds)

    ground_truth_path = lines_root / "gt.json"
    LOG.info("Saving ground truth.")
    save_ground_truth_json(
        gt, ground_truth_path,
        gt_formats={GTFormat.JSON, GTFormat.TESSERACT}
    )

    LOG.info("Done.")
    LOG.info(f"    Total samples checked: {filter.total_loaded}")
    LOG.info(f"    Total amount of skipped words: {filter.num_skipped}")
