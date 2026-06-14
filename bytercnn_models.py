import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# def byte_rcnn_model(maxlen, embed_dim, RNN_SIZE, CNN_SIZE, kernels, output_cnt, OUTPUT_PATH, initial_learning_rate=0.001):

#     inputs = layers.Input(shape=(maxlen,))

#     # FIX 1: correct embedding
#     emb = layers.Embedding(256, embed_dim)(inputs)

#     x_context = layers.Dropout(0.3)(emb)
#     x_context = layers.Bidirectional(layers.GRU(RNN_SIZE, return_sequences=True))(x_context)
#     x_context = layers.Bidirectional(layers.GRU(RNN_SIZE, return_sequences=True))(x_context)

#     x = layers.Concatenate()([emb, x_context])

#     # FIX 2: proper conv pipeline
#     convs = []
#     for k in kernels:
#         c = layers.Conv1D(CNN_SIZE, k, activation='relu')(x)
#         c = layers.MaxPool1D(4)(c)
#         c = layers.Conv1D(CNN_SIZE, k, activation='relu')(c)
#         convs.append(c)

#     poolings = (
#         [layers.GlobalAveragePooling1D()(c) for c in convs] +
#         [layers.GlobalMaxPooling1D()(c) for c in convs]
#     )

#     x = layers.Concatenate()(poolings)

#     # FIX 3: stronger regularization
#     x = layers.Dense(512, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)

#     x = layers.Dense(256, activation="relu")(x)
#     x = layers.Dropout(0.5)(x)

#     outputs = layers.Dense(output_cnt, activation="softmax")(x)

#     model = keras.Model(inputs=inputs, outputs=outputs)

#     opt = tf.keras.optimizers.Adam(
#         learning_rate=initial_learning_rate,
#         beta_1=0.9,
#         beta_2=0.98,
#         epsilon=1e-09,
#         amsgrad=True
#     )

#     model.compile(
#         optimizer=opt,
#         loss="sparse_categorical_crossentropy",
#         metrics=["accuracy"]
#     )

#     return model


def byte_rcnn_model(maxlen, embed_dim, RNN_SIZE, CNN_SIZE, kernels, output_cnt, OUTPUT_PATH, initial_learning_rate=0.001):

    inputs = layers.Input(shape=(maxlen,))
    emb = layers.Embedding(256, embed_dim)(inputs)

    # DEEPER RNN
    x = layers.Bidirectional(
        layers.GRU(RNN_SIZE, return_sequences=True)
    )(emb)

    x = layers.Bidirectional(
        layers.GRU(RNN_SIZE, return_sequences=True)
    )(x)

    # CNN on top
    convs = []
    for k in kernels:
        c = layers.Conv1D(CNN_SIZE, k, activation='relu')(x)
        c = layers.MaxPooling1D(2)(c)
        convs.append(c)

    x = layers.Concatenate()(
        [layers.GlobalMaxPooling1D()(c) for c in convs] +
        [layers.GlobalAveragePooling1D()(c) for c in convs]
    )

    x = layers.Dense(384, activation='relu')(x)
    x = layers.Dropout(0.4)(x)

    outputs = layers.Dense(output_cnt, activation='softmax')(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model