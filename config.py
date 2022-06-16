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
