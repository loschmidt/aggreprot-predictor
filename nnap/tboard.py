import tensorflow as tf
from tensorflow.keras.callbacks import Callback

from .metrics import calculate_metrics


class TBMetricCallback(Callback):
    def __init__(self, log_dir, x, y):
        super().__init__()
        self.x = x
        self.y = y
        self.writer = tf.summary.create_file_writer(log_dir + "/metrics")

    def on_epoch_end(self, epoch, logs=None):
        y_ = self.model.predict(self.x)
        metrics = calculate_metrics(self.y, y_)
        with self.writer.as_default():
            for m, v in metrics.items():
                tf.summary.scalar(m, v, step=epoch)
