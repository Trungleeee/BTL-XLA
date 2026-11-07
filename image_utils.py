from collections import deque

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def load_rgb(source):
    """Đọc ảnh, sửa hướng EXIF và ghép nền trong suốt lên nền trắng."""
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGBA")

    background = Image.new("RGBA", image.size, "white")
    return Image.alpha_composite(background, image).convert("RGB")


def otsu_threshold(gray):
    """
    Tính ngưỡng nhị phân tự động theo thuật toán Otsu.
    Cực đại hóa phương sai giữa hai lớp nền và đối tượng.
    """
    hist, _ = np.histogram(gray, bins=256, range=(0, 256))
    total = gray.size

    if total == 0:
        return 127

    sum_total = float(np.dot(np.arange(256), hist))
    sum_bg = 0.0
    weight_bg = 0
    max_variance = -1.0
    threshold = 127

    for t in range(256):
        weight_bg += int(hist[t])
        if weight_bg == 0:
            continue

        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += float(t * hist[t])
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = (
            float(weight_bg)
            * float(weight_fg)
            * (mean_bg - mean_fg) ** 2
        )

        if variance > max_variance:
            max_variance = variance
            threshold = t

    return int(np.clip(threshold, 30, 220))


def binary_mask(gray):
    """Tạo mặt nạ nhị phân đối tượng bằng ngưỡng Otsu."""
    if int(gray.max()) < 30:
        return np.zeros_like(gray, dtype=bool)

    threshold = otsu_threshold(gray)
    return gray > threshold


def foreground_array(image):
    """
    Trả về ảnh xám uint8 đã khử nền và tăng cường tương phản:
    - nền tối (0)
    - đối tượng sáng (0..255)

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
        border = 255 - border

    # Khử mức nền dư (bóng đổ nhẹ, giấy ngả màu) và kéo dãn tương phản.
    bg_level = float(np.percentile(border, 75))
    shifted = np.clip(gray.astype(np.float32) - bg_level, 0.0, 255.0)
    peak = float(shifted.max())

    if peak < 15.0:
        return np.zeros_like(gray)

    stretched = (shifted / peak) * 255.0
    return np.clip(np.round(stretched), 0, 255).astype(np.uint8)


def _center_by_mass(canvas, resized, size):
    """
    Đặt ảnh đã co giãn vào khung vuông sao cho trọng tâm khối lượng
    (Center of Mass) nằm gần tâm khung ảnh nhất (chuẩn MNIST),
    nhưng không làm cắt mất biên đối tượng.
    """
    new_width, new_height = resized.size
    arr = np.asarray(resized, dtype=np.float32)
    total_mass = float(arr.sum())

    default_x = (size - new_width) // 2
    default_y = (size - new_height) // 2

    if total_mass <= 1e-6:
        canvas.paste(resized, (default_x, default_y))
        return

    ys, xs = np.indices(arr.shape)
    cy = float((ys * arr).sum() / total_mass)
    cx = float((xs * arr).sum() / total_mass)

    target = (size - 1) / 2.0
    paste_x = round(target - cx)
    paste_y = round(target - cy)

    # Giữ toàn bộ đối tượng nằm trọn trong khung [0, size].
    paste_x = int(np.clip(paste_x, 0, max(0, size - new_width)))
    paste_y = int(np.clip(paste_y, 0, max(0, size - new_height)))

    canvas.paste(resized, (paste_x, paste_y))


def normalize_object(
    gray,
    size,
    margin,
    use_center_of_mass=False,
    dilate_thin=False,
):
    """
    Cắt vùng có đối tượng bằng ngưỡng Otsu, giữ tỷ lệ khi resize,
    đặt vào giữa ảnh vuông và chuẩn hóa về [0, 1].

    Kết quả có shape: (size, size, 1).
    """
    mask = binary_mask(gray)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise ValueError("Không tìm thấy đối tượng rõ trong ảnh.")

    crop = gray[
        ys.min():ys.max() + 1,
        xs.min():xs.max() + 1
    ]

    height, width = crop.shape
    crop_image = Image.fromarray(crop)

    # Làm dày nét tự động nếu ảnh gốc độ phân giải lớn mà nét bút quá mảnh.
    if dilate_thin and max(height, width) > size * 2:
        stroke_ratio = float((crop > otsu_threshold(crop)).mean())
        if stroke_ratio < 0.14:
            crop_image = crop_image.filter(ImageFilter.MaxFilter(size=3))

    available = size - 2 * margin
    scale = available / max(height, width)

    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))

    resized = crop_image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("L", (size, size), 0)

    if use_center_of_mass:
        _center_by_mass(canvas, resized, size)
    else:
        canvas.paste(
            resized,
            (
                (size - new_width) // 2,
                (size - new_height) // 2,
            ),
        )

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    if array.max() > 0:
        array = array / float(array.max())

    return array[..., np.newaxis]


def prepare_digit(image):
    gray = foreground_array(image)
    normalized = normalize_object(
        gray,
        size=28,
        margin=4,
        use_center_of_mass=True,
        dilate_thin=True,
    )
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
    mask = binary_mask(gray)
    regions = connected_components(mask, min_area)

    objects = []

    for region in regions:
        x0, y0, x1, y1 = region["bbox"]
        box_w = max(1, x1 - x0)
        box_h = max(1, y1 - y0)
        area = len(region["pixels"])

        # Chỉ lấy pixel thuộc vùng đang xét.
        crop = np.zeros((box_h, box_w), dtype=np.uint8)

        for y, x in region["pixels"]:
            crop[y - y0, x - x0] = gray[y, x]

        normalized = normalize_object(crop, size=64, margin=6)

        objects.append({
            "bbox": region["bbox"],
            "image": normalized,
            "area": area,
            "aspect_ratio": round(box_w / box_h, 2),
            "extent": round(area / (box_w * box_h), 3),
        })

    return mask, objects