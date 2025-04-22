import os
import json

from glob import glob
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tqdm import tqdm


def find_best_model_by(given_metric: str, experiment_name: str, model_type: str, save_model_name: str = None) -> Model:
    path = os.path.join(experiment_name, "hsearch", "*", "*.json")
    best_metric = 0
    best_file = ""
    if not save_model_name.endswith(".hdf5"):
        save_model_name = f"{save_model_name}.hdf5"

    if save_model_name:
        best_model_save_path = os.path.join(experiment_name, "best_model")
        os.makedirs(best_model_save_path, exist_ok=True)

    for file in tqdm(glob(path, recursive=True)):
        with open(file, "r") as fp:
            metric = json.load(fp)
            metric = float(metric['metrics'][given_metric]) if len(metric['metrics']) > 0 else None
            if metric == None:
                continue
            if metric > best_metric:
                best_metric = metric
                best_file = file

    print(f"best models metric: {best_metric}")

    model_file = os.path.join(experiment_name, os.path.basename(os.path.dirname(best_file)), model_type, "fold_0",
                              "models", given_metric, f"model.{round(best_metric, 2)}.hdf5")
    model = load_model(model_file) if os.path.isfile(model_file) else exit(1)
    model.summary()
    if save_model_name:

        model.save(os.path.join(best_model_save_path))
        with open(os.path.join(best_model_save_path, given_metric), "w") as fp:
            fp.write(str(best_metric))
    else:
        return model, best_metric


if __name__ == "__main__":
    metric = "val_auc"
    experiment_name = '../hsearch'
    model_type = "sequential"
    save_model = "model"
    find_best_model_by(metric, experiment_name, model_type, save_model)
