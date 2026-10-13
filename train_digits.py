import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from networks import build_cnn
from training import train_and_evaluate


def main():
    keras.utils.set_random_seed(42)

    (x_train, y_train), (x_test, y_test) = (
        keras.datasets.mnist.load_data()
    )

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.1,
        random_state=42,
        stratify=y_train,
    )

    model = build_cnn(
        input_shape=(28, 28, 1),
        num_classes=10,
    )

    train_and_evaluate(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        name="digit_cnn",
        class_names=[str(i) for i in range(10)],
    )


if __name__ == "__main__":
    main()