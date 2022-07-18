import argparse
from pathlib import Path

from utils.dataset_utils import add_dataset_args, parse_dataset_args, preprocess_dataset, load_ground_truth_json, \
    dataset_to_ground_truth, save_ground_truth_json, GTFormat


def register(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-dest",
        required=True,
        help="Destination directory where preprocessed items and ground truth will be stored"
    )
    parser.add_argument(
        "-keep",
        action="store_true",
        help="Use cached items instead of making new ones"
    )
    parser.add_argument(
        "-full",
        action="store_true",
        help="Also include blurred and adaptive thresholded (and may be more)."
    )
    add_dataset_args(parser)


def handle(args: argparse.Namespace):

    dest = Path(args.dest)
    dest.mkdir(exist_ok=True)

    ds, voc = parse_dataset_args(args)
    preprocess_dataset(
        ds, only_threshold=False,
        cache_dir=str(dest),
        keep=args.keep,
        full=args.full
    )

    gt_path = dest / "gt.json"
    stored_gt = load_ground_truth_json(gt_path) if gt_path.exists() else {}
    stored_gt.update(dataset_to_ground_truth(ds))
    save_ground_truth_json(
        stored_gt, gt_path,
        gt_formats={GTFormat.JSON, GTFormat.TESSERACT}
    )
