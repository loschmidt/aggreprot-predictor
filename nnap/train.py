import dataclasses
import logging
import os
from enum import Enum
from typing import Dict, List

from kitt.environment import write_yaml
from kitt.experiment.experiment import ExperimentTracker, generate_name_from_params
from kitt.experiment.experiment import Run, Parameter
from kitt.gpu.utils.tf import clear_gpu_memory
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tqdm import tqdm

from nnap.data import (
    parse_data,
    get_folds_no,
    data_formater,
    folds_processor,
)
from nnap.metrics import process_training_logs
from nnap.models import create_model
from nnap.tboard import TBMetricCallback


class TrainingType(Enum):
    Simple = "simple"
    CV = "cv"


@dataclasses.dataclass
class TrainConfig:
    epochs: int
    model_type: str
    batch_size: int
    dropout: float
    units: int
    rnn_layers: int
    dense_layers: int
    learning_rate: float


def train(
        train_data, config: TrainConfig, static_file, seq_file, out_dir, training_type
):
    training_type = TrainingType(training_type)

    if config.model_type == "sequential":
        proteins_t, x1_t, x2_t, y_t, folds = parse_data(train_data, None, seq_file)
        folds_no = get_folds_no(folds)
        x = data_formater(x1_t, folds, folds_no)
        y = data_formater(y_t, folds, folds_no)
        input_shape = (x[0].shape[1:],)

    elif config.model_type == "static":
        proteins_t, x1_t, x2_t, y_t, folds = parse_data(train_data, static_file, None)
        folds_no = get_folds_no(folds)
        x = data_formater(x2_t, folds, folds_no)
        y = data_formater(y_t, folds, folds_no)
        input_shape = (x[0].shape[1:],)

    else:
        proteins_t, x1_t, x2_t, y_t, folds = parse_data(
            train_data, static_file, seq_file
        )
        folds_no = get_folds_no(folds)
        x1_t = data_formater(x1_t, folds, folds_no)
        x2_t = data_formater(x2_t, folds, folds_no)
        x = (x1_t, x2_t)
        y = data_formater(y_t, folds, folds_no)
        input_shape = (x1_t[0].shape[1:], x2_t[0].shape[1:])

    data_splits = folds_processor(x, y, folds_no)
    training_logs = []
    for index, data_split in enumerate(data_splits):
        model = create_model(config, input_shape)
        model.summary()
        train, val = data_split
        log_dir = os.path.join(out_dir, config.model_type, f"fold_{index}", "logs")
        cp_path = os.path.join(out_dir, config.model_type, f"fold_{index}", "models")
        tb = TensorBoard(
            log_dir=log_dir,
            write_graph=True,
            update_freq="epoch",
        )
        checkpoint_val_loss = ModelCheckpoint(
            os.path.join(cp_path, "val_loss", "model.val_loss.hdf5"),
            save_best_only=True,
            mode="min",
        )
        checkpoint_tp = ModelCheckpoint(
            os.path.join(cp_path, "tp", "model.tp.hdf5"),
            save_best_only=True,
            monitor="tp",
            mode="max",
        )
        checkpoint_tn = ModelCheckpoint(
            os.path.join(cp_path, "tn", "model.tn.hdf5"),
            save_best_only=True,
            monitor="tn",
            mode="max",
        )
        checkpoint_fn = ModelCheckpoint(
            os.path.join(cp_path, "fn", "model.fn.hdf5"),
            save_best_only=True,
            monitor="fn",
            mode="min",
        )
        checkpoint_val_auc = ModelCheckpoint(
            os.path.join(cp_path, "val_auc", "model.val_auc.hdf5"),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
        )
        checkpoint_fp = ModelCheckpoint(
            os.path.join(cp_path, "fp", "model.fp.hdf5"),
            save_best_only=True,
            monitor="fp",
            mode="min",
        )
        checkpoint_auc = ModelCheckpoint(
            os.path.join(cp_path, "auc", "model.auc.hdf5"),
            save_best_only=True,
            monitor="auc",
            mode="max",
        )

        tb_metric_cb = TBMetricCallback(log_dir, val[0], val[1])

        history = model.fit(
            train[0],
            train[1],
            validation_data=(val[0], val[1]),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=[
                tb,
                checkpoint_val_loss,
                checkpoint_tp,
                checkpoint_tn,
                checkpoint_fn,
                checkpoint_fp,
                checkpoint_auc,
                tb_metric_cb,
                checkpoint_val_auc
            ],
        )
        training_logs.append(history)

        if training_type == TrainingType.Simple:
            break

    stats = process_training_logs(training_logs)
    print(stats)
    return training_logs


def train_hsearch(
        config: TrainConfig,
        train_data: str,
        static_file: str,
        seq_file: str,
        data_dir: str,
) -> Dict:
    histories = train(train_data, config, static_file, seq_file, data_dir, "simple")

    clear_gpu_memory()
    metrics = {}
    for i, history in enumerate(histories):
        for metric, values in history.history.items():
            # TODO: Combine metrics across folds
            metrics[f"{metric}_last_{i}"] = values[-1]
            metrics[f"{metric}_min_{i}"] = min(values)
            metrics[f"{metric}_max_{i}"] = max(values)

    return metrics


def run_experiments(
        experiment_name: str,
        configurations: List[TrainConfig],
        train_data: str,
        data_dir: str,
        static_file: str,
        sequence_file: str,
):
    experiment = ExperimentTracker(experiment_name, data_dir)

    for (index, config) in tqdm(enumerate(configurations), total=len(configurations)):
        logging.info(f"Starting configuration: {config}")

        """
        hparams = {
            "epochs": Parameter(value=config.epochs, is_hyperparameter=True, value_str=str(config.epochs)),
            "model_type": Parameter(value=config.model_type, is_hyperparameter=True, value_str=str(config.model_type)),
            "batch_size": Parameter(value=config.batch_size, is_hyperparameter=True, value_str=str(config.batch_size)),
            "dropout": Parameter(value=config.dropout, is_hyperparameter=True, value_str=str(config.dropout)),
            "units": Parameter(value=config.units, is_hyperparameter=True, value_str=str(config.units)),
            "rnn_layers": Parameter(value=config.rnn_layers, is_hyperparameter=True, value_str=str(config.rnn_layers)),
            "dense_layers": Parameter(value=config.dense_layers, is_hyperparameter=True, value_str=str(config.dense_layers)),
            "learning_rate": Parameter(value=config.learning_rate, is_hyperparameter=True, value_str=str(config.learning_rate))
        }
        """

        hparams = {
            "epochs": config.epochs,
            "model_type": config.model_type,
            "batch_size": config.batch_size,
            "dropout": config.dropout,
            "units": config.units,
            "rnn_layers": config.rnn_layers,
            "dense_layers": config.dense_layers,
            "learning_rate": config.learning_rate,
        }

        name = generate_name_from_params(hparams)

        with experiment.new_run(name=name) as run:
            with open(run.data_path("config.yml"), "w") as f:
                write_yaml(config, f)

            for key, value in hparams.items():
                run.record_param(key, value)

            #run.record_params(hparams)

            metrics = train_hsearch(
                config,
                train_data,
                static_file,
                sequence_file,
                os.path.join(data_dir, name),
            )

            for key, value in metrics.items():
                run.record_metric(key, value)
