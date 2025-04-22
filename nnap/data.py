import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing import sequence
from typing import Union, List, Optional
from numpy.typing import NDArray

LABELIZER = {"non-amyloid": 0.0, "amyloid": 1.0}
SEQ_LEN = 6


class HexapeptidesIO:
    data: pd.DataFrame

    def __init__(self, hexapeptides_or_path: Union[List[str], str]):
        if type(hexapeptides_or_path) is str:
            self.data = pd.read_csv(hexapeptides_or_path, header=None, sep=";")
        elif type(hexapeptides_or_path) is list:
            self.data = pd.DataFrame(hexapeptides_or_path)
        else:
            raise Exception

    def save(self, out_file: str):
        with open(out_file, "w") as file:
            self.data.to_csv(file, header=False, index=False, sep=";")

    def to_pd(self):
        return pd.DataFrame(self.data)


def foldify(no_folds: int, samples: pd.DataFrame) -> NDArray:
    folds = [i % no_folds for i in range(samples.count()[0])]
    np.random.shuffle(folds)
    return folds


def get_other_folds(array: NDArray, index: int) -> NDArray:
    output = []
    for i in range(len(array)):
        output.append(array[i]) if i != index else None
    output = np.concatenate(output)
    return output


def folds_processor(x: NDArray, y: NDArray, folds_count: int):
    if isinstance(x, tuple):
        x1, x2 = x
    else:
        x1 = x
        x2 = None

    for fold_i in range(folds_count):
        val = (
            ((x1[fold_i], x2[fold_i]), y[fold_i])
            if isinstance(x2, np.ndarray)
            else (x[fold_i], y[fold_i])
        )

        train = (
            (
                (get_other_folds(x1, fold_i), get_other_folds(x2, fold_i)),
                get_other_folds(y, fold_i),
            )
            if isinstance(x2, np.ndarray)
            else (get_other_folds(x, fold_i), get_other_folds(y, fold_i))
        )

        yield train, val


def labelize(str_label: str) -> float:
    return LABELIZER[str_label.lower()] if str_label.lower() in LABELIZER else str_label


def parse_static_file(file: str) -> dict:
    static_features = {}
    with open(file, "r", encoding="utf-8-sig") as f:
        for l in f.readlines():
            try:
                items = l.rstrip().split(";")
                static_features[items[0]] = [float(f) for f in items[1:]]
            except:
                print(f"Failed to parse {l}")
    return static_features


def parse_seq_file(file: str) -> dict:
    acid_features = {}
    with open(file, "r", encoding="utf-8-sig") as f:
        for l in f.readlines():
            items = l.rstrip().replace(",", ".").split(";")
            acid_features[items[0]] = [float(f) for f in items[1:]]
    return acid_features


def get_folds_no(datafile: NDArray) -> int:
    return len(set(datafile))


def data_formater(
        array: NDArray, shuffle_indexes: NDArray, folds_no: int
) -> NDArray:
    output = []
    for fold in range(folds_no):
        output.append(array[np.where(shuffle_indexes == fold)])
    return np.array(output)


def parse_sample_file(file: str, want_labelize=True) -> pd.DataFrame:
    samples = pd.read_csv(file, sep=";", names=["hexpeptide", "label", "fold_id"])
    samples["label"] = (
        samples["label"].apply(labelize) if want_labelize else samples["label"]
    )
    return samples


def parse_data(data_or_data_file: Union[str, pd.DataFrame], static_file: Optional[str], seq_file: Optional[str], want_labelize=True) -> tuple:
    proteins = []
    x1 = []
    x2 = []
    y = []
    folds = []
    if isinstance(data_or_data_file, str):
        samples = parse_sample_file(data_or_data_file, want_labelize)
    else:
        samples = data_or_data_file
    static_features = parse_static_file(static_file) if static_file else None
    seq_features = parse_seq_file(seq_file) if seq_file else None

    for data in samples.itertuples(False):
        protein, label, fold_id = list(data)
        if len(protein) != SEQ_LEN:
            print(f"Skipping protein {protein}, MAX_LEN set to {SEQ_LEN}")
            continue
        try:
            x1_ = (
                np.asarray([seq_features[acid] for acid in protein])
                if seq_features
                else None
            )
            x2_ = static_features[protein] if static_features else None

            proteins.append(protein)
            x1.append(x1_)
            x2.append(x2_)
            y.append(label)
            folds.append(fold_id)
        except Exception as e:
            print(f"Failed to process {protein}: {e}")

    if seq_features:
        x1 = sequence.pad_sequences(np.array(x1), maxlen=SEQ_LEN)
    return proteins, np.array(x1), np.array(x2), np.array(y), np.array(folds)
