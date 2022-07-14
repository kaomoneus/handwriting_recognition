"""
Default cache directory value
"""
import os
from pathlib import Path


"""
Defines internal format of rendered strings.
Note, if string is quite short for such aspect ration
it is supposed to pad such rendered string image.
"""
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 64
PAD_COLOR = 255


"""
Training batch size
"""
BATCH_SIZE_DEFAULT = 256

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
TRAIN_VALIDATE_CNT = 16*64

"""
Default number of train epochs
"""
TRAIN_EPOCHS_DEFAULT = 10

"""
Dataset shuffler seed. It's important to keep this seed same each time
you learn *same* model. This seed is used for initial dataset shuffling.

Splitting dataset onto train.train, train.validate and test sets will happen
afterwards.
"""
DATASET_SHUFFLER_SEED = 1

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
Default path to marked list
"""
MARKED_PATH_DEFAULT = ".marked.json"


"""
Default path to whitelist (generated whenever dataset is loaded)
"""
WHITELIST_PATH_DEFAULT = ".last_whitelist.json"


"""
Default path to tensorboard logs dir
"""
TENSORBOARD_LOGS_DEFAULT = ".tensorboard_logs"

"""
Amount of rows for ploti command
"""
PLOTI_ROWS = 8

"""
Amount of cols for ploti command
"""
PLOTI_COLS = 8
