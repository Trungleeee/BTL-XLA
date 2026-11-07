import json
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import ImageDraw
from tensorflow import keras

from image_utils import (
    extract_shapes,
    load_rgb,
    otsu_threshold,
    prepare_digit,
)


ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Nhận dạng ảnh bằng CNN",
    layout="wide",
)


@st.cache_resource
def load_model(path):
    return keras.models.load_model(path, compile=False)


def get_model(filename):
    path = ROOT / "models" / filename

    if not path.exists():
        st.error(
            f"Chưa tìm thấy {filename}. "
            "Hãy chạy file huấn luyện tương ứng trước."
        )
        st.stop()

    return load_model(str(path))


def show_features(model, sample):
    """Hiển thị 8 bản đồ đặc trưng của lớp tích chập đầu."""
    feature_model = keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("conv1").output,
    )

    features = feature_model.predict(
        sample[np.newaxis, ...],
        verbose=0,
    )[0]

    st.write("Bản đồ đặc trưng của lớp Conv2D đầu tiên")

    columns = st.columns(4)

    for index in range(8):
        feature = features[:, :, index]
        value_range = float(feature.max() - feature.min())

        display = (
            (feature - feature.min()) / max(value_range, 1e-8)
        )

        columns[index % 4].image(
            display,
            caption=f"Kênh {index + 1}",
            clamp=True,
            width=140,
        )


def digit_page():
    st.header("Nhận dạng chữ số viết tay")
    st.caption(
        "Mỗi ảnh chứa một chữ số 0–9. Hệ thống tự động khử nền, "
        "phân ngưỡng Otsu và căn chỉnh theo trọng tâm khối lượng (Center of Mass)."
    )

    model = get_model("digit_cnn.keras")

    uploaded = st.file_uploader(
        "Chọn ảnh chữ số",
        type=["png", "jpg", "jpeg"],
        key="digit_upload",
    )

    if uploaded is None:
        return

    try:
        image = load_rgb(uploaded)
        image.thumbnail((800, 800))
        gray, normalized = prepare_digit(image)
        threshold = otsu_threshold(gray)
    except Exception as error:
        st.error(f"Không xử lý được ảnh: {error}")
        return

    columns = st.columns(3)

    columns[0].image(image, caption="Ảnh gốc")
    columns[1].image(
        gray,
        caption=f"Khử nền & Tăng tương phản (Ngưỡng Otsu = {threshold})",
    )
    columns[2].image(
        normalized[:, :, 0],
        caption="Ảnh 28×28 căn tâm khối lượng đưa vào CNN",
        clamp=True,
        width=200,
    )

    probabilities = model.predict(
        normalized[np.newaxis, ...],
        verbose=0,
    )[0]

    predicted = int(np.argmax(probabilities))

    st.success(f"Chữ số dự đoán: {predicted}")
    st.write(f"Điểm dự đoán: {probabilities[predicted]:.2%}")

    st.bar_chart({
        "Điểm dự đoán": probabilities.tolist()
    })

    with st.expander("Xem bản đồ đặc trưng"):
        show_features(model, normalized)


def shape_page():
    st.header("Phát hiện hình tròn và hình chữ nhật")
    st.caption(
        "Phân đoạn tự động bằng ngưỡng Otsu + BFS 8 lân cận. "
        "Có thể thử samples/multiple_shapes.png."
    )

    model = get_model("shape_cnn.keras")

    uploaded = st.file_uploader(
        "Chọn ảnh hình học",
        type=["png", "jpg", "jpeg"],
        key="shape_upload",
    )

    col_area, col_conf = st.columns(2)

    min_area = col_area.slider(
        "Số pixel tối thiểu của một vùng",
        min_value=5,
        max_value=300,
        value=30,
    )

    min_confidence = col_conf.slider(
        "Ngưỡng độ tin cậy tối thiểu",
        min_value=0.50,
        max_value=0.99,
        value=0.75,
        step=0.05,
    )

    if uploaded is None:
        return

    try:
        image = load_rgb(uploaded)

        # Giới hạn kích thước để BFS chạy nhanh hơn.
        image.thumbnail((800, 800))

        mask, objects = extract_shapes(
            image,
            min_area=min_area,
        )
    except Exception as error:
        st.error(f"Không xử lý được ảnh: {error}")
        return

    left, right = st.columns(2)
    left.image(image, caption="Ảnh đầu vào")
    right.image(
        mask.astype(np.uint8) * 255,
        caption="Ảnh nhị phân (Phân ngưỡng tự động Otsu)",
    )

    if not objects:
        st.warning("Không tìm thấy vùng hình phù hợp.")
        return

    batch = np.stack([obj["image"] for obj in objects])
    probabilities = model.predict(batch, verbose=0)

    result = image.copy()
    draw = ImageDraw.Draw(result)

    names = ["Tròn", "Chữ nhật"]
    ascii_names = ["Circle", "Rectangle"]
    colors = ["#00a651", "#e63946"]
    counts = [0, 0, 0]
    rows = []

    for index, (obj, scores) in enumerate(
        zip(objects, probabilities), start=1
    ):
        label = int(np.argmax(scores))
        confidence = float(scores[label])
        x0, y0, x1, y1 = obj["bbox"]

        if confidence >= min_confidence:
            counts[label] += 1
            pred_name = names[label]
            tag = f"{index}: {ascii_names[label]} ({confidence:.0%})"
            box_color = colors[label]
        else:
            counts[2] += 1
            pred_name = "Chưa rõ / Nhiễu"
            tag = f"{index}: Uncertain ({confidence:.0%})"
            box_color = "#f4a261"

        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=box_color,
            width=3,
        )

        draw.text(
            (x0, max(0, y0 - 14)),
            tag,
            fill=box_color,
        )

        rows.append({
            "Vùng": index,
            "Dự đoán": pred_name,
            "Điểm tin cậy": round(confidence, 4),
            "Diện tích (px)": obj.get("area", 0),
            "Tỷ lệ W/H": obj.get("aspect_ratio", 1.0),
            "Độ đặc (Extent)": obj.get("extent", 0.0),
            "Khung bao": str(obj["bbox"]),
        })

    st.image(result, caption="Kết quả phát hiện (Xanh lá: Tròn, Đỏ: Chữ nhật, Cam: Chưa rõ)")
    st.write(
        f"**Hình tròn**: {counts[0]} — "
        f"**Hình chữ nhật**: {counts[1]} — "
        f"**Chưa rõ / Dưới ngưỡng**: {counts[2]}"
    )
    st.dataframe(rows, use_container_width=True)

    with st.expander("Xem ảnh cắt và đặc trưng"):
        choice = st.selectbox(
            "Chọn vùng",
            options=list(range(len(objects))),
            format_func=lambda value: f"Vùng {value + 1}",
        )

        sample = objects[choice]["image"]

        st.image(
            sample[:, :, 0],
            caption="Ảnh 64×64 đưa vào CNN",
            clamp=True,
            width=200,
        )

        show_features(model, sample)


def report_page():
    st.header("Kết quả huấn luyện")

    name = st.selectbox(
        "Chọn mô hình",
        ["digit_cnn", "shape_cnn"],
    )

    folder = ROOT / "reports" / name

    if not folder.exists():
        st.info("Chưa có báo cáo. Hãy huấn luyện mô hình trước.")
        return

    metrics_path = folder / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        c1, c2 = st.columns(2)
        c1.metric(
            "Độ chính xác tập Test (Test Accuracy)",
            f"{metrics['test_accuracy']:.2%}",
        )
        c2.metric(
            "Hàm mất mát tập Test (Test Loss)",
            f"{metrics['test_loss']:.4f}",
        )

    for filename in ["training.png", "confusion_matrix.png"]:
        path = folder / filename
        if path.exists():
            st.image(str(path))

    report_path = folder / "classification_report.txt"

    if report_path.exists():
        st.code(report_path.read_text(encoding="utf-8"))


st.title("Nhận dạng chữ số và hình học bằng CNN")

page = st.sidebar.radio(
    "Chức năng",
    [
        "Chữ số viết tay",
        "Hình học",
        "Báo cáo huấn luyện",
    ],
)

if page == "Chữ số viết tay":
    digit_page()
elif page == "Hình học":
    shape_page()
else:
    report_page()