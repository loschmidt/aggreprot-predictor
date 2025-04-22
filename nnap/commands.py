import os
import re
import itertools
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tabulate import tabulate
from tensorflow.keras.models import load_model
from tqdm import tqdm

from nnap.data import (
    parse_data,
    foldify,
    HexapeptidesIO,
)
from nnap.metrics import calculate_metrics
from nnap.models import get_model_type


def split_data(output_file, data_file, no_folds):
    # samples = parse_sample_file(data_file, want_labelize=False)
    samples = HexapeptidesIO(data_file).to_pd()
    # Shuffle

    samples["fold_id"] = foldify(no_folds, samples)
    samples.to_csv(output_file, sep=";", header=False)


def predict(model_file: str, data_file: str, static_file: Optional[str], seq_file: Optional[str], out_file: str, want_metrics: bool = False):
    model = load_model(model_file)
    model_type = get_model_type(model)
    model.summary()
    if model_type == "sequential":
        proteins, x1, x2, y, folds = parse_data(data_file, None, seq_file, want_labelize=want_metrics)
        y_ = model.predict(x1)
    elif model_type == "static":
        proteins, x1, x2, y, folds = parse_data(data_file, static_file, None, want_labelize=want_metrics)
        y_ = model.predict(x2)
    else:
        proteins, x1, x2, y, folds = parse_data(data_file, static_file, seq_file, want_labelize=want_metrics)
        y_ = model.predict((x1, x2))

    # saving prediction info
    dirname = os.path.dirname(out_file)
    if dirname != "":
        os.makedirs(dirname, exist_ok=True)
    with open(out_file, "w") as f:
        lines = [
            f"{protein},{prediction}\n"
            for protein, prediction in zip(proteins, y_.flatten())
        ]
        f.writelines(lines)

    if want_metrics:
        calculated_metrics = calculate_metrics(y, y_)
        print(calculated_metrics)
        return calculated_metrics

def create_profile(prediction_file: str, out_file: str, aggregation_func=np.mean):
    predictions = []
    protein = ""
    with open(prediction_file, "r") as fp:
        hexapeptide = ""
        for l in fp:
            hexapeptide, value = l.rstrip().split(",")
            predictions.append(float(value))
            protein += hexapeptide[0]
        protein += hexapeptide[1:]
    profile = calculate_profile(protein, predictions, aggregation_func)
    with open(out_file, "w") as fp:
        for residue, value in zip(protein, profile):
            fp.write(f"{residue},{value}\n")

def calculate_profile(protein, predictions, aggregation_func=np.mean):
    window_size = 6
    profile = []
    for residue_idx in range(1, len(protein) + 1):
        start_idx = max(residue_idx - window_size, 0)
        stop_idx = residue_idx
        pred_slice = predictions[start_idx:stop_idx]
        value = aggregation_func(pred_slice)
        profile.append(value)
    assert len(protein) == len(profile)
    return profile

METRICS_TRAIN_NAME = ["val_loss", "auc", "fn", "fp", "tn", "tp"]


def compare(model_path, data_file, static_file, seq_file, out_file):
    # TODO?: save table into csv?? or to another format??
    datatable = pd.DataFrame()

    # iterate over folders and save models path, models metric type, model step save
    for path in tqdm(glob(os.path.join(model_path, "**", "*.hdf5"), recursive=True)):
        metric = os.path.basename(os.path.dirname(path))
        model_name = os.path.basename(path)
        regex = re.findall("[0-9]+", model_name)
        epoch = int(regex[0])
        metric_value = float(re.findall("([0-9]+([.][0-9]*))", model_name)[0][0])
        datatable = datatable.append(
            {metric: model_name, "path": path, "epochs": epoch, "value": metric_value},
            ignore_index=True,
        )

    best_models = {}
    # iterate over metrics and find best model for every metric
    for metric in tqdm(METRICS_TRAIN_NAME):
        datarow = (
            datatable[[metric, "epochs", "path", "value"]]
            .sort_values(by="epochs", ascending=False)
            .dropna()
            .head(1)
        )
        metric_head = f"{metric} ({datarow['value'].values[0]})"
        best_models[metric_head] = datarow["path"].values[0]

    # assert the right number of models
    assert len(best_models) == len(METRICS_TRAIN_NAME)

    # visuallization helper variable
    pandas_columns = [
        *METRICS_TRAIN_NAME,
        "sensitivity",
        "specificity",
        "accuracy",
        "q_value",
        "f1_score",
        "mcc",
    ]

    # pretty dataframe initialization
    datatable = pd.DataFrame(index=best_models.keys(), columns=pandas_columns)
    # compute metrics and save into pandas datatable
    for metric in tqdm(best_models):
        path = best_models[metric]
        # predict on test data
        metrics = predict(
            path,
            data_file,
            static_file,
            seq_file,
            os.path.join(os.path.dirname(out_file), "data_compare", metric, out_file),
        )
        # put into table
        datatable.loc[metric] = metrics

    # format and print pretty table
    datatable.index.name = "Models training metric"
    print(tabulate(datatable, "keys", tablefmt="psql"))


class MultipleModelBatchPredictor:
    def __init__(self, models_dir, static_file: Optional[str], seq_file: Optional[str], window_size: int = 6):
        model_paths = glob(f"{models_dir}/*.hdf5")
        self.model_names = []
        self.models = []
        self.model_types = []
        for model_file in model_paths:
            model_name = Path(model_file).stem
            self.model_names.append(model_name)
            model = load_model(model_file)
            self.models.append(model)
            self.model_types.append(get_model_type(model))
        self.static_file = static_file
        self.seq_file = seq_file
        self.window_size = window_size

    def predict(self, fasta_file, out_dir, delimiter: str = '\t', want_metrics=False):
        from Bio import SeqIO
        import csv
        parser_handle = SeqIO.parse(fasta_file, "fasta")

        os.makedirs(out_dir, exist_ok=True)
        mean_out_path = os.path.join(out_dir, f"batch-profile-final-mean.txt")
        max_out_path = os.path.join(out_dir, f"batch-profile-final-max.txt")
        min_out_path = os.path.join(out_dir, f"batch-profile-final-min.txt")

        with open(mean_out_path, "w") as fo_mean, open(max_out_path, "w") as fo_max, open(min_out_path, "w") as fo_min:
            wr_mean = csv.writer(fo_mean, delimiter=delimiter)
            wr_max = csv.writer(fo_max, delimiter=delimiter)
            wr_min = csv.writer(fo_min, delimiter=delimiter)
            for record in parser_handle:
                hexapeptides = [
                    str(record.seq[i: i + self.window_size])
                    for i in range(len(record.seq) - self.window_size + 1)
                ]
                data = parse_data(pd.DataFrame({"hexapeptides": hexapeptides, "label": None, "fold_id": None}), self.static_file, self.seq_file, want_labelize=want_metrics)

                predictions = []
                for model, model_type in zip(self.models, self.model_types):
                    if model_type == "sequential":
                        proteins, x1, x2, y, folds = data
                        y_ = model.predict(x1)
                    elif model_type == "static":
                        proteins, x1, x2, y, folds = data
                        y_ = model.predict(x2)
                    else:
                        proteins, x1, x2, y, folds = data
                        y_ = model.predict((x1, x2))
                    predictions.append(y_)
                predictions = np.asarray(predictions)
                model_mean = np.mean(predictions, axis=0).reshape((-1,))
                final_mean = calculate_profile(record.seq, model_mean, np.mean)
                final_min = calculate_profile(record.seq, model_mean, np.min)
                final_max = calculate_profile(record.seq, model_mean, np.max)
                wr_mean.writerow(itertools.chain([record.id, record.seq], final_mean))
                wr_min.writerow(itertools.chain([record.id, record.seq], final_min))
                wr_max.writerow(itertools.chain([record.id, record.seq], final_max))
