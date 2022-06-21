"""
Default cache directory value
"""
import os
from pathlib import Path


"""
Training batch size
"""
BATCH_SIZE = 64

"""
Defines proportion of training and test data
Training data includes:
* training samples itself - samples we train model on
* validate samples - samples we evaluate progress during training
   passed as additional parameter into 'fit' method.
Test data is used after training is finished.
"""
TRAIN_TEST_RATIO = 0.9

"""
Defines (absolute) amount of validation samples (used in 'fit' method)
"""
TRAIN_VALIDATE_CNT = 100*BATCH_SIZE

"""
Default number of train epochs
"""
TRAIN_EPOCHS_DEFAULT = 10

"""
Default path to preprocessed cache dir
"""
CACHE_DIR_DEFAULT = Path(os.environ["HOME"]) / ".handwritten_preprocessing_cache"

"""
Default list of words to be ignored
"""
# Currently we put signs which require additional alignment
# and as long we don't implement such alignment we keep them here
TRAIN_IGNORE_LIST_DEFAULT = [
    ",", ".", "#", "/", "`", "'", '"', "'", "M0", "M", "0M", "OM"
]

"""
Default maximum word length, if used, then all longer samples will be skipped
"""
MAX_WORD_LEN_DEFAULT = 21

"""
Default LSTM vocabulary
"""
VOCABULARY_DEFAULT = {
    "characters": "-'./0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "max_len": MAX_WORD_LEN_DEFAULT,
    "ignore": TRAIN_IGNORE_LIST_DEFAULT
}

"""
Amount of rows for ploti command
"""
PLOTI_ROWS = 8

"""
Amount of cols for ploti command
"""
PLOTI_COLS = 8
