import glob
import itertools

import click
import numpy as np
import pandas as pd
from pathlib import Path

from nnap import STATIC_FILE, SEQ_FILE, SEQ_MODEL_DIR
from nnap.commands import *
from nnap.find_best_model import find_best_model_by
from nnap.data import HexapeptidesIO
from nnap.sov import SOV_measure_i


@click.command(name="split-data", short_help="dataset creator")
@click.argument("output_file")
@click.argument("data_file")
@click.option("--no-folds", default=5, type=int)
def split_data_cmd(output_file, data_file, no_folds):
    """
    method to create dataset \n
    create the dataset with data selection training/validation and folds selection
    """
    split_data(output_file, data_file, no_folds)


@click.command(name="train")
@click.argument("train_data")
@click.option("--epochs", default=1000)
@click.option(
    "--model-type",
    default="combined",
    type=click.Choice(["combined", "sequential", "static"]),
)
@click.option("--static-file", default=STATIC_FILE)
@click.option("--seq-file", default=SEQ_FILE)
@click.option("--out-dir", default=".")
@click.option(
    "--training-type",
    default="simple",
    type=click.Choice(["simple", "cv"]),
)
@click.option("--batch-size", default=16, type=int)
@click.option("--dropout", default=0.0, type=float)
@click.option("--units", default=512, type=int)
@click.option("--rnn-layers", default=2, type=int)
@click.option("--dense-layers", default=1, type=int)
@click.option("--learning-rate", default=0.001, type=float)
def train_cmd(
    train_data, epochs, model_type, static_file, seq_file, out_dir, training_type,
    batch_size, dropout, units, rnn_layers, dense_layers, learning_rate
):
    from nnap.train import train, TrainConfig

    config = TrainConfig(
        epochs, model_type, batch_size, dropout,
        units, rnn_layers, dense_layers, learning_rate
    )
    # train(train_data, epochs, model_type, static_file, seq_file, out_dir, training_type)
    train(train_data, config, static_file, seq_file, out_dir, training_type)


@click.command(name="predict", short_help="predictor")
@click.argument("model-file")
@click.argument("data-file")
@click.option("--static-file", default=STATIC_FILE)
@click.option("--seq-file", default=SEQ_FILE)
@click.option("--out-file", default="out.csv")
def predict_cmd(model_file, data_file, static_file, seq_file, out_file):
    predict(model_file, data_file, static_file, seq_file, out_file, False)


@click.command(name="predict-dir", short_help="predictor")
@click.argument("model-dir")
@click.argument("data-dir")
@click.option("--static-file", default=STATIC_FILE)
@click.option("--seq-file", default=SEQ_FILE)
@click.option("--out-dir", default="out")
def predict_dir_cmd(model_dir, data_dir, static_file, seq_file, out_dir):
    predict_dir(model_dir, data_dir, static_file, seq_file, out_dir)

def predict_dir(model_dir, data_dir, static_file, seq_file, out_dir):
    model_paths = glob(f"{model_dir}/*.hdf5")
    for data_file in glob(f"{data_dir}/hexapeptides-*.txt"):
        data_name = Path(data_file).stem
        out_files = []
        df = pd.DataFrame()
        model_names = []
        for model_file in model_paths:
            model_name = Path(model_file).stem
            model_names.append(model_name)
            prediction_file = os.path.join(out_dir, f"prediction-{data_name}-{model_name}.txt")
            out_files.append(prediction_file)
            predict(model_file, data_file, static_file, seq_file, prediction_file, False)
            create_profile(prediction_file, os.path.join(out_dir, f"profile-{data_name}-{model_name}.txt"))
            d = pd.read_csv(prediction_file, header=None)
            df[model_name] = d[1]
            hexapeptides = d[0]
        df["hexapeptides"] = hexapeptides
        df["model_mean"] = df[[mn for mn in model_names]].mean(axis=1)
        mean_profile_file = os.path.join(out_dir, f"profile-model-mean-{data_name}.txt")
        df[["hexapeptides", "model_mean"]].to_csv(mean_profile_file, index=False, header=None)
        create_profile(mean_profile_file, os.path.join(out_dir, f"profile-final-mean-{data_name}.txt"), np.mean)
        create_profile(mean_profile_file, os.path.join(out_dir, f"profile-final-max-{data_name}.txt"), np.max)
        create_profile(mean_profile_file, os.path.join(out_dir, f"profile-final-min-{data_name}.txt"), np.min)


@click.command(name="predict-sequential", short_help="Split protein sequence and predict")
@click.argument("protein-file")
@click.option("--fasta", is_flag=True, default=False)
@click.option("--out-dir", default="out")
def predict_sequential_cmd(protein_file, fasta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    hexapeptides_file = os.path.join(out_dir, "hexapeptides.txt")
    if fasta:
        from Bio import SeqIO
        record = SeqIO.read(protein_file, "fasta")
        split_protein_str(str(record.seq), hexapeptides_file)
    else:
        split_protein(protein_file, hexapeptides_file)
    model_paths = glob(f"{SEQ_MODEL_DIR}/*.hdf5")
    df = pd.DataFrame()
    model_names = []
    hexapeptides = None
    for model_file in model_paths:
        model_name = Path(model_file).stem
        model_names.append(model_name)
        prediction_file = os.path.join(out_dir, f"prediction-{model_name}.txt")
        predict(model_file, hexapeptides_file, None, SEQ_FILE, prediction_file, False)
        create_profile(prediction_file, os.path.join(out_dir, f"profile-{model_name}.txt"))
        d = pd.read_csv(prediction_file, header=None)
        df[model_name] = d[1]
        hexapeptides = d[0]
    assert hexapeptides is not None
    df["hexapeptides"] = hexapeptides
    df["model_mean"] = df[[mn for mn in model_names]].mean(axis=1)
    mean_profile_file = os.path.join(out_dir, f"profile-model-mean.txt")
    df[["hexapeptides", "model_mean"]].to_csv(mean_profile_file, index=False, header=None)
    create_profile(mean_profile_file, os.path.join(out_dir, f"profile-final-mean.txt"), np.mean)
    create_profile(mean_profile_file, os.path.join(out_dir, f"profile-final-max.txt"), np.max)
    create_profile(mean_profile_file, os.path.join(out_dir, f"profile-final-min.txt"), np.min)

@click.command(name="predict-sequential-batch", short_help="Predict multiple sequences from single FASTA file.")
@click.argument("fasta-file")
@click.option("-o", "--out-dir", default="out", show_default=True, help="Output directory.")
@click.option("-d", "--delimiter", default="\t", help="Use specified delimiter instead of TAB.")
def predict_sequential_batch_cmd(fasta_file, out_dir, delimiter):
    assert len(delimiter) > 0, 'Delimiter cannot be empty.'
    pred = MultipleModelBatchPredictor(SEQ_MODEL_DIR, None, SEQ_FILE, window_size=6)
    pred.predict(fasta_file, out_dir, delimiter=delimiter, want_metrics=False)


@click.command(name="compare")
@click.argument("model-path")
@click.argument("data-file")
@click.option("--static-file", default=STATIC_FILE)
@click.option("--seq-file", default=SEQ_FILE)
@click.option("--out-file", default="out.csv")
def compare_cmd(model_path, data_file, static_file, seq_file, out_file):
    compare(model_path, data_file, static_file, seq_file, out_file)


@click.command(name="multiple-split-predict")
@click.argument("proteins_path")
@click.argument("model_path")
@click.option("--out-path", default="./")
def multiple_split_predict_cmd(proteins_path, model_path, out_path):
    for protein_path in os.listdir(proteins_path):
        if os.path.isfile(os.path.join(proteins_path, protein_path)):
            file_name = os.path.splitext(protein_path)[0]
            mk_dir = out_path + file_name
            if not os.path.exists(mk_dir):
                os.mkdir(mk_dir)
            split_protein(proteins_path + '/' + protein_path, out_path + file_name + '/' + 'hexapeptides-' + file_name + '.txt')
            predict_dir(model_path, mk_dir, 'data/waltzdb_export.csv', 'data/Atomic.csv', mk_dir)

@click.command(name="split-protein")
@click.argument("protein-file")
@click.option("--out-file", default="hexapeptides.fasta")
def split_protein_cmd(protein_file, out_file):
    split_protein(protein_file, out_file)

def split_protein(protein_file, out_file):
    with open(protein_file, "r") as f:
        protein_sequence = f.read().rstrip()
    return split_protein_str(protein_sequence, out_file)

def split_protein_str(protein_sequence, out_file):
    window_size = 6
    hexapeptides = [
        protein_sequence[i : i + window_size]
        for i in range(len(protein_sequence) - window_size + 1)
    ]

    fasta = HexapeptidesIO(hexapeptides)
    fasta.save(out_file)


@click.command(name="create-profile")
@click.argument("prediction-file")
@click.option("--out-file", default="profile.csv")
def create_profile_cmd(prediction_file, out_file):
    create_profile(prediction_file, out_file)


@click.command(name="hsearch")
@click.argument("data-file")
@click.option("--static-file", default=STATIC_FILE)
@click.option("--seq-file", default=SEQ_FILE)
@click.option("--out-dir", default="out")
@click.option("--experiment-name", default="hsearch")
@click.option("--units")
def hsearch_cmd(data_file, static_file, seq_file, out_dir, experiment_name, units):
    dropout_values = [dr / 10.0 for dr in range(0, 10, 2)]
    unit_values = [int(units)]
    #unit_values = [8, 16, 32, 64, 128, 256, 512]
    rnn_counts = [1, 2]
    dense_counts = [1, 2, 3]
    epochs = [100]
    batch_sizes = [16, 32]
    model_types = ["combined"]
    learning_rates = [0.0001, 0.001, 0.01, 0.1]

    params = itertools.product(
        epochs,
        model_types,
        batch_sizes,
        dropout_values,
        unit_values,
        rnn_counts,
        dense_counts,
        learning_rates,
    )
    from nnap.train import run_experiments, TrainConfig

    configurations = [TrainConfig(*p) for p in params]

    run_experiments(
        experiment_name,
        configurations,
        data_file,
        out_dir,
        static_file,
        seq_file,
    )

    """" FIXME: What is this?
    model, metric = find_best_model_by("val_auc", experiment_name, "sequential")
    model.summary()
    print(f"best model has {round(metric, 3)}")
    """


@click.command(name="find-best-model")
@click.argument("searched-metric")
@click.argument("experiment-name")
@click.option(
    "--model-type",
    default="sequential",
    type=click.Choice(["combined", "sequential", "static"]),
)
@click.option("--saved-model-name", default="model")
def find_best_model_cmd(metric, experiment_name, model_type, save_model_name):
    """
    Function to search, localize and copy best model into given folder
    """
    find_best_model_by(metric, experiment_name, model_type, save_model_name)


def measure_core(data_file):
    data = pd.read_csv(data_file, sep=';', names=["key", "pred_num", "tresholded", "real"])
    predicted = data['tresholded'].to_list()
    predicted = [str(i) for i in predicted]
    predicted = np.array(predicted)

    real = data['real'].to_list()
    real = [str(i) for i in real]
    real = np.array(real)


    N_i, fractions, lenSGti = SOV_measure_i('0', real, predicted)
    SOV_measure = sum(fractions) / N_i
    res_0 = round(SOV_measure * 100, 3)
    print(f"SOV measure for 0 is {res_0}%")

    N_i, fractions, lenSGti = SOV_measure_i('1', real, predicted)
    SOV_measure = sum(fractions) / N_i
    res_1 = round(SOV_measure * 100, 3)
    print(f"SOV measure for 1 is {res_1}%")
    return res_0, res_1


@click.command()
@click.argument("data-file")
def measure(data_file):
    measure_core(data_file)

@click.group()
def cli():
    """
    Aminoacid amyloid AI classification program.
    """
    pass


def main():
    cli.add_command(train_cmd)
    cli.add_command(predict_cmd)
    cli.add_command(multiple_split_predict_cmd)
    cli.add_command(split_protein_cmd)
    cli.add_command(create_profile_cmd)
    cli.add_command(split_data_cmd)
    cli.add_command(compare_cmd)
    cli.add_command(hsearch_cmd)
    cli.add_command(find_best_model_cmd)
    cli.add_command(predict_dir_cmd)
    cli.add_command(predict_sequential_cmd)
    cli.add_command(predict_sequential_batch_cmd)
    cli.add_command(measure)
    cli()


if __name__ == "__main__":
    main()
