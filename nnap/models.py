import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import LSTM, Dropout, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(config, input_shapes):
    if config.model_type == "sequential":
        model = model_seq(config, input_shapes[0])
    elif config.model_type == "static":
        model = model_static(config, input_shapes[0])
    elif config.model_type == "combined":
        model = model_combined(config, input_shapes[0], input_shapes[1])
    else:
        raise Exception(f"Unsupported model type {config.model_type}")
    return model


def get_model_type(model: Model) -> str:
    input_layers = [layer for layer in model.layers if type(layer) == InputLayer]
    if len(input_layers) == 2:
        return "combined"
    elif len(input_layers[0].input_shape[0][1:]) == 2:
        return "sequential"
    elif len(input_layers[0].input_shape[0][1:]) == 1:
        return "static"
    else:
        raise Exception("Unsupported model type")


def model_combined(config, seq_input_shape, static_input_shape) -> Model:
    seq_layers = [Input(shape=seq_input_shape, name="seq_input")]
    for i in range(config.rnn_layers):
        seq_layers.append(
            Bidirectional(
                LSTM(
                    config.units,
                    dropout=config.dropout,
                    return_sequences=False if i == config.rnn_layers - 1 else True,
                )
            )(seq_layers[-1])
        )

    static_layers = [Input(shape=static_input_shape, name="static_input")]
    for _ in range(config.dense_layers):
        static_layers.append(
            Dropout(config.dropout)(Dense(config.units, activation="relu")(static_layers[-1]))
        )

    concat = concatenate([seq_layers[-1], static_layers[-1]], name="concatenate")
    dense = Dropout(config.dropout)(Dense(config.units, activation="relu")(concat))
    model_output = Dense(1, activation="sigmoid")(dense)
    model = Model(
        inputs=[seq_layers[0], static_layers[0]],
        outputs=model_output,
        name="final_output",
    )
    return compile_model(model, config.learning_rate)


def model_seq(config, seq_input_shape) -> Model:
    layers = [Input(shape=seq_input_shape, name="seq_input")]
    for i in range(config.rnn_layers):
        layers.append(
            Bidirectional(
                LSTM(
                    config.units,
                    dropout=config.dropout,
                    return_sequences=False if i == config.rnn_layers - 1 else True,
                )
            )(layers[-1])
        )
    for _ in range(config.dense_layers):
        layers.append(
            Dropout(config.dropout)(Dense(config.units, activation="relu")(layers[-1]))
        )

    layers.append(Dense(1, activation="sigmoid")(layers[-1]))

    model = Model(inputs=[layers[0]], outputs=layers[-1], name="final_output")
    return compile_model(model, config.learning_rate)


def model_static(config, static_input_shape) -> Model:
    static_layers = [Input(shape=static_input_shape, name="static_input")]
    for _ in range(config.dense_layers):
        static_layers.append(
            Dropout(config.dropout)(Dense(config.units, activation="relu")(static_layers[-1]))
        )
    final_model_output = Dense(1, activation="sigmoid")(static_layers[-1])
    model = Model(inputs=static_layers[0], outputs=final_model_output, name="final_output")
    return compile_model(model, config.learning_rate)


def compile_model(model, learning_rate) -> Model:
    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(learning_rate=learning_rate),
        metrics=[
            tf.keras.metrics.TruePositives(name="tp"),
            tf.keras.metrics.FalsePositives(name="fp"),
            tf.keras.metrics.TrueNegatives(name="tn"),
            tf.keras.metrics.FalseNegatives(name="fn"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model
