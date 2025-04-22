import os

__version__ = "1.0.2.3-alpha.0"

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/data/"

STATIC_FILE = DATA_DIR + "waltzdb_export.csv"

SEQ_FILE = DATA_DIR + "Atomic.csv"
SEQ_MODEL_DIR = DATA_DIR + "models/sequential"
