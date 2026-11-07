from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape, num_classes, augment=False):
    inputs = keras.Input(shape=input_shape, name="input_image")
    x = inputs

    if augment:
        x = layers.RandomRotation(0.04, fill_mode="constant")(x)
        x = layers.RandomZoom(0.05, fill_mode="constant")(x)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv1",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv2",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        activation="relu",
        name="conv3",
    )(x)
    x = layers.BatchNormalization()(x)

    if input_shape[0] >= 64:
        x = layers.MaxPooling2D(pool_size=2)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model