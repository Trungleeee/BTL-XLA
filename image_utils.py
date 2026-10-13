from collections import deque

import numpy as np
from PIL import Image, ImageOps


def load_rgb(source):
    """Đọc ảnh, sửa hướng EXIF và ghép nền trong suốt lên nền trắng."""
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")

    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def foreground_array(image):
    """
    Trả về ảnh xám uint8:
    - nền tối
    - đối tượng sáng

    Giả định vùng sát mép ảnh chủ yếu là nền.
    """
    gray = np.asarray(image.convert("L"), dtype=np.uint8)

    # Ảnh gần như đồng màu: xem là không có đối tượng.
    if int(gray.max()) - int(gray.min()) < 20:
        return np.zeros_like(gray)

    border = np.concatenate([
        gray[0, :],
        gray[-1, :],
        gray[:, 0],
        gray[:, -1],
    ])

    if np.median(border) > 127:
        gray = 255 - gray

    return gray


def normalize_object(gray, size, margin):
    """
    Cắt vùng có đối tượng, giữ tỷ lệ khi resize,
    đặt vào giữa ảnh vuông và chuẩn hóa về [0, 1].

    Kết quả có shape: (size, size, 1).
    """
    mask = gray > 127
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Không tìm thấy đối tượng rõ trong ảnh.")

    crop = gray[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    height, width = crop.shape
    available = size - 2 * margin
    scale = available / max(height, width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = Image.fromarray(crop).resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (size, size), 0)
    canvas.paste(
        resized,
        (
            (size - new_width) // 2,
            (size - new_height) // 2,
        ),
    )

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return array[..., np.newaxis]


def prepare_digit(image):
    gray = foreground_array(image)
    normalized = normalize_object(gray, size=28, margin=4)
    return gray, normalized


def connected_components(mask, min_area=30):
    """
    BFS với 8 pixel lân cận.

    Mỗi vùng trả về:
    - bbox: (x0, y0, x1, y1), x1/y1 không bao gồm
    - pixels: các tọa độ (y, x) thuộc vùng
    """
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    regions = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue

            queue = deque([(y, x)])
            visited[y, x] = True
            pixels = []

            x_min = x_max = x
            y_min = y_max = y

            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))

                x_min = min(x_min, cx)
                x_max = max(x_max, cx)
                y_min = min(y_min, cy)
                y_max = max(y_max, cy)

                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue

                        ny = cy + dy
                        nx = cx + dx

                        if not (0 <= ny < height and 0 <= nx < width):
                            continue

                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((ny, nx))

            if len(pixels) >= min_area:
                regions.append({
                    "bbox": (x_min, y_min, x_max + 1, y_max + 1),
                    "pixels": pixels,
                })

    return regions


def extract_shapes(image, min_area=30):
    gray = foreground_array(image)
    mask = gray > 127
    regions = connected_components(mask, min_area)

    objects = []

    for region in regions:
        x0, y0, x1, y1 = region["bbox"]

        # Chỉ lấy pixel thuộc vùng đang xét.
        crop = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)

        for y, x in region["pixels"]:
            crop[y - y0, x - x0] = gray[y, x]

        normalized = normalize_object(crop, size=64, margin=6)

        objects.append({
            "bbox": region["bbox"],
            "image": normalized,
        })

    return mask, objects