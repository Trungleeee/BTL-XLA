import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
)
from tensorflow import keras


ROOT = Path(__file__).resolve().parent


def train_and_evaluate(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    x_test,
    y_test,
    name,
    class_names,
):
    models_dir = ROOT / "models"
    report_dir = ROOT / "reports" / name

    models_dir.mkdir(exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=15,
        batch_size=64,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    model.save(models_dir / f"{name}.keras")

    test_loss, test_accuracy = model.evaluate(
        x_test, y_test, verbose=0
    )

    probabilities = model.predict(x_test, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    labels = list(range(len(class_names)))

    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )

    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print(report)

    (report_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    metrics = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "class_names": class_names,
    }

    (report_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    history_data = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    (report_dir / "history.json").write_text(
        json.dumps(history_data, indent=2),
        encoding="utf-8",
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, metric in zip(axes, ["accuracy", "loss"]):
        ax.plot(history.history[metric], label="Train")
        ax.plot(history.history[f"val_{metric}"], label="Validation")
        ax.set_title(metric)
        ax.set_xlabel("Epoch index")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(report_dir / "training.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        labels=labels,
        display_labels=class_names,
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    fig.tight_layout()
    fig.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Đã lưu mô hình: models/{name}.keras")
    print(f"Đã lưu báo cáo: reports/{name}")