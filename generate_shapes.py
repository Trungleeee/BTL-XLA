from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from image_utils import foreground_array, normalize_object


ROOT = Path(__file__).resolve().parent
rng = np.random.default_rng(42)


def create_shape(label):
    image = Image.new("L", (128, 128), 255)
    draw = ImageDraw.Draw(image)

    cx = int(rng.integers(56, 73))
    cy = int(rng.integers(56, 73))

    if label == 0:
        width = height = int(rng.integers(30, 77))
    else:
        width = int(rng.integers(30, 77))
        height = int(rng.integers(24, 77))

    box = (
        cx - width // 2,
        cy - height // 2,
        cx + width // 2,
        cy + height // 2,
    )

    color = int(rng.integers(0, 81))
    stroke_width = int(rng.integers(2, 6))
    filled = bool(rng.integers(0, 2))

    options = {
        "fill": color if filled else None,
        "outline": color,
        "width": stroke_width,
    }

    if label == 0:
        draw.ellipse(box, **options)
    else:
        draw.rectangle(box, **options)

    image = image.rotate(
        float(rng.uniform(-45, 45)),
        resample=Image.Resampling.BICUBIC,
        fillcolor=255,
    )

    return image


def main():
    data_dir = ROOT / "data"
    samples_dir = ROOT / "samples"

    data_dir.mkdir(exist_ok=True)
    samples_dir.mkdir(exist_ok=True)

    images = []
    labels = []

    for label in (0, 1):
        for index in range(3000):
            image = create_shape(label)

            normalized = normalize_object(
                foreground_array(image),
                size=64,
                margin=6,
            )

            images.append(normalized)
            labels.append(label)

            if index < 5:
                image.save(samples_dir / f"class_{label}_{index}.png")

    x = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)

    np.savez_compressed(
        data_dir / "shapes.npz",
        x=x,
        y=y,
    )

    # Ảnh demo gồm hai hình tròn và hai hình chữ nhật.
    demo = Image.new("RGB", (640, 400), "white")
    draw = ImageDraw.Draw(demo)

    draw.ellipse((40, 40, 170, 170), outline="black", width=5)
    draw.rectangle((240, 50, 420, 160), outline="black", width=5)
    draw.ellipse((70, 230, 190, 350), fill="black")
    draw.rectangle((320, 230, 550, 340), fill="black")

    demo.save(samples_dir / "multiple_shapes.png")

    print("Đã tạo dữ liệu:", x.shape)
    print("Nhãn: 0 = circle, 1 = rectangle")
    print("Ảnh demo: samples/multiple_shapes.png")


if __name__ == "__main__":
    main()