import argparse
import json
import logging
from typing import Dict, Any, List

import tensorflow as tf
from keras.layers import StringLookup

from config import VOCABULARY_DEFAULT

LOG = logging.getLogger(__name__)
PADDING_TOKEN = 99


# TODO: combine it with Model
#   instead "model" in context of our task is a combination of keras.Model + vocabulary.
#   Thus it should be built together, load/save together
class Vocabulary:
    def __init__(self, characters: str = None, max_len: int = None, ignore: List[str] = None):
        # Pay attention that "set" is mandatory to be transformed to sorted "list"
        # Otherwise whenever you start new python session and restore your model from storage
        # you'll get "mad" neural network, just because it uses vocabulary with
        # different characters order.
        self.characters = list(characters) if characters else None
        self.max_len = max_len
        self.num_to_char = None
        self.char_to_num = None
        self.ignore = ignore
        if self.characters:
            self.build_convertors()

    def build_convertors(self):
        # Mapping characters to integers.
        self.char_to_num = StringLookup(vocabulary=list(self.characters), mask_token=None)

        # Mapping integers back to original characters.
        self.num_to_char = StringLookup(
            vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True
        )

    class Builder:
        def __init__(self, subj):
            # TODO: (only) in builder also store characters frequency.
            self.characters = None
            self.max_len = 0
            self.subj = subj

        def __enter__(self):
            self.characters = set()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if not self.characters:
                return
            self.subj.characters = "".join(sorted(self.characters))
            self.subj.max_len = self.max_len
            self.subj.build_convertors()

            # FIXME: in original article max_len calculated only for train dataset.
            LOG.info(f"Maximum discovered word length: {self.max_len}")
            LOG.info(f"Discovered vocab size: {len(self.characters)}")

        def update(self, label: str):
            self.max_len = max(self.max_len, len(label))
            self.characters.update(label)

    def builder(self):
        return Vocabulary.Builder(self)

    def vectorize_label(self, label):
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=PADDING_TOKEN)
        return label

    def save(self, path: str):
        with open(path, "w") as f:
            fields = dict(
                characters="".join(self.characters),
                max_len=self.max_len,
                ignore=self.ignore
            )
            json.dump(fields, f, indent=4)


def load_vocabulary(path: str) -> Vocabulary():
    with open(path, "r") as f:
        voc = json.load(f)
        return Vocabulary(**voc)


def add_voc_args(parser: argparse.ArgumentParser):
    voc_group = parser.add_mutually_exclusive_group()

    voc_group.add_argument(
        "-voc",
        dest="vocabulary",
        help="Path to custom vocabulary file."
             " If vocabulary is specified, then"
             " all words with character not present in"
             " vocabulary will be ignored.",
    )
    voc_group.add_argument(
        "-voc-auto",
        dest="voc_auto",
        help="Generates vocabulary from dataset",
        action="store_true"
    )
    voc_group.add_argument(
        "-no-voc",
        dest="no_voc",
        help="Voc is disabled, load None",
        action="store_true"
    )


def parse_voc_args(args: argparse.Namespace):
    if args.voc_auto:
        LOG.info("Using auto vocabulary")
        return None
    if args.no_voc:
        LOG.info("Vocabulary disabled")
        return None
    if not args.vocabulary:
        LOG.info("Using default vocabulary")
        return Vocabulary(**VOCABULARY_DEFAULT)

    LOG.info("Using vocabulary: " + args.vocabulary)
    return load_vocabulary(args.vocabulary)
