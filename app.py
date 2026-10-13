from pathlib import Path

import numpy as np
import streamlit as st
from PIL import ImageDraw
from tensorflow import keras

from image_utils import load_rgb, prepare_digit, extract_shapes


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
    st.caption("Mỗi ảnh chứa một chữ số 0–9, nền sạch.")

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
    except Exception as error:
        st.error(f"Không xử lý được ảnh: {error}")
        return

    columns = st.columns(3)

    columns[0].image(image, caption="Ảnh gốc")
    columns[1].image(gray, caption="Nét sáng trên nền tối")
    columns[2].image(
        normalized[:, :, 0],
        caption="Ảnh 28×28 đưa vào CNN",
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
        "Dùng ảnh nền đơn giản, hình tách rời và tương phản rõ. "
        "Có thể thử samples/multiple_shapes.png."
    )

    model = get_model("shape_cnn.keras")

    uploaded = st.file_uploader(
        "Chọn ảnh hình học",
        type=["png", "jpg", "jpeg"],
        key="shape_upload",
    )

    min_area = st.slider(
        "Số pixel tối thiểu của một vùng",
        min_value=5,
        max_value=300,
        value=30,
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
        caption="Ảnh nhị phân",
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
    counts = [0, 0]
    rows = []

    for index, (obj, scores) in enumerate(
        zip(objects, probabilities), start=1
    ):
        label = int(np.argmax(scores))
        counts[label] += 1

        x0, y0, x1, y1 = obj["bbox"]

        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline="red",
            width=3,
        )

        draw.text(
            (x0, max(0, y0 - 14)),
            f"{index}: {ascii_names[label]}",
            fill="red",
        )

        rows.append({
            "Vùng": index,
            "Dự đoán": names[label],
            "Điểm": round(float(scores[label]), 4),
            "Khung bao": str(obj["bbox"]),
        })

    st.image(result, caption="Kết quả phát hiện")
    st.write(
        f"Hình tròn: {counts[0]} — "
        f"Hình chữ nhật: {counts[1]}"
    )
    st.dataframe(rows)

    st.caption(
        "Mô hình hiện chỉ có hai lớp. Hình ngoài phạm vi "
        "vẫn có thể bị gán thành tròn hoặc chữ nhật."
    )

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