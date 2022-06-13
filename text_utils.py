import logging
import tensorflow as tf

LOG = logging.getLogger(__name__)
PADDING_TOKEN = 99


class Vocabulary:
    def __init__(self):
        # Pay attention that "set" is mandatory to be transformed to sorted "list"
        # Otherwise whenever you start new python session and restore your model from storage
        # you'll get "mad" neural network, just because it uses vocabulary with
        # different characters order.
        self.characters = None
        self.max_len = 0
        self.num_to_char = None
        self.char_to_num = None

    class Loader:
        def __init__(self, subj):
            self.characters = None
            self.max_len = 0
            self.subj = subj

        def __enter__(self):
            self.characters = set()
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            from keras.layers import StringLookup

            self.subj.characters = list(sorted(self.characters))

            # Mapping characters to integers.
            self.subj.char_to_num = StringLookup(vocabulary=list(self.subj.characters), mask_token=None)

            # Mapping integers back to original characters.
            self.subj.num_to_char = StringLookup(
                vocabulary=self.subj.char_to_num.get_vocabulary(), mask_token=None, invert=True
            )

            self.subj.max_len = self.max_len

            # FIXME: in original article max_len calculated only for train dataset.
            LOG.info(f"Maximum length: {self.max_len}")
            LOG.info(f"Vocab size: {len(self.characters)}")

        def update(self, label: str):
            self.max_len = max(self.max_len, len(label))
            self.characters.update(label)

    def loader(self):
        return Vocabulary.Loader(self)

    def vectorize_label(self, label):
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=PADDING_TOKEN)
        return label
