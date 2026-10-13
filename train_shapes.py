from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

from networks import build_cnn
from training import train_and_evaluate


ROOT = Path(__file__).resolve().parent


def main():
    keras.utils.set_random_seed(42)

    dataset_path = ROOT / "data" / "shapes.npz"

    if not dataset_path.exists():
        raise FileNotFoundError(
            "Chưa có dữ liệu. Hãy chạy generate_shapes.py trước."
        )

    with np.load(dataset_path) as data:
        x = data["x"]
        y = data["y"]

    # 70% train, 30% tạm giữ lại.
    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    )

    # Chia phần 30% thành 15% validation và 15% test.
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=42,
        stratify=y_temp,
    )

    model = build_cnn(
        input_shape=(64, 64, 1),
        num_classes=2,
    )

    train_and_evaluate(
        model,
        x_train,
        y_train,
        x_val,
        y_val,
        x_test,
        y_test,
        name="shape_cnn",
        class_names=["circle", "rectangle"],
    )


if __name__ == "__main__":
    main()