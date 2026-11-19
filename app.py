
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
import time
import io


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fruit Detection | YOLOv8",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #0e1117;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 2.5rem;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #aab4c3;
    font-size: 1rem;
    margin-bottom: 25px;
}

.metric-card {
    background-color: #1c2333;
    border: 1px solid #303b50;
    padding: 18px;
    border-radius: 12px;
    text-align: center;
}

.metric-label {
    color: #aab4c3;
    font-size: 0.9rem;
}

.metric-value {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 700;
}

.section-title {
    color: #ffffff;
    font-size: 1.3rem;
    font-weight: 700;
}

div[data-testid="stMetric"] {
    background-color: #1c2333;
    border: 1px solid #303b50;
    padding: 15px;
    border-radius: 10px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "weights" / "best.pt"

CLASS_NAMES = [
    "Apple",
    "Avocado",
    "Banana",
    "Mango",
    "Orange"
]


# =========================================================
# LOAD YOLO MODEL
# =========================================================

@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🍎 Fruit Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered fruit detection using a custom-trained YOLOv8 model.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# CHECK MODEL
# =========================================================

if not MODEL_PATH.exists():
    st.error("Model file not found!")
    st.write(f"Expected model location: {MODEL_PATH}")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as e:
    st.error(f"Could not load YOLO model: {e}")
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("⚙️ Detection Settings")

st.sidebar.markdown("### Confidence Threshold")

confidence = st.sidebar.slider(
    "Confidence",
    min_value=0.05,
    max_value=0.95,
    value=0.25,
    step=0.05
)

st.sidebar.markdown("### IoU Threshold")

iou_threshold = st.sidebar.slider(
    "IoU",
    min_value=0.10,
    max_value=0.90,
    value=0.45,
    step=0.05
)

st.sidebar.markdown("### Image Size")

image_size = st.sidebar.select_slider(
    "Inference Resolution",
    options=[320, 416, 512, 640, 800, 1024],
    value=640
)

st.sidebar.divider()

st.sidebar.markdown("### Model Information")

st.sidebar.write("**Architecture:** YOLOv8 Nano")
st.sidebar.write("**Training Epochs:** 20")
st.sidebar.write("**Classes:** 5")
st.sidebar.write("**Task:** Object Detection")

st.sidebar.divider()

st.sidebar.markdown("### Supported Fruits")

for fruit in CLASS_NAMES:
    st.sidebar.write(f"🍏 {fruit}")


# =========================================================
# MODEL STATUS
# =========================================================

st.success("YOLOv8 model loaded successfully!")


# =========================================================
# MODEL OVERVIEW
# =========================================================

st.markdown("### 📊 Model Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Model", "YOLOv8n")

with col2:
    st.metric("Classes", "5")

with col3:
    st.metric("Epochs", "20")

with col4:
    st.metric("Input Size", f"{image_size}px")


# =========================================================
# IMAGE UPLOAD
# =========================================================

st.divider()

st.markdown("### 📤 Upload Fruit Image")

uploaded_file = st.file_uploader(
    "Choose an image containing fruits",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG, JPEG, or PNG image."
)


# =========================================================
# IMAGE INFERENCE
# =========================================================

if uploaded_file is not None:

    try:
        # Read image using PIL and convert to RGB
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("### 🖼️ Uploaded Image")

        # Keep original PIL image for inference
        col1, col2 = st.columns(2)

        with col1:
            st.image(
                image,
                caption="Original Image",
                use_container_width=True
            )

        # Run YOLO prediction directly on PIL image
        start_time = time.perf_counter()

        results = model.predict(
            source=image,
            conf=confidence,
            iou=iou_threshold,
            imgsz=image_size,
            device="cpu",
            verbose=False
        )

        processing_time = time.perf_counter() - start_time

        result = results[0]

        # Get annotated image
        # Ultralytics plot() returns BGR NumPy image
        annotated_image = result.plot()

        # Convert BGR to RGB for Streamlit display
        annotated_image = annotated_image[:, :, ::-1]

        with col2:
            st.image(
                annotated_image,
                caption="YOLOv8 Detection Results",
                use_container_width=True
            )

        # =================================================
        # EXTRACT DETECTION INFORMATION
        # =================================================

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0].item())

                score = float(box.conf[0].item())

                class_name = result.names[class_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detections.append({
                    "Fruit": class_name,
                    "Confidence": round(score * 100, 2),
                    "X1": round(x1, 2),
                    "Y1": round(y1, 2),
                    "X2": round(x2, 2),
                    "Y2": round(y2, 2)
                })

        df = pd.DataFrame(detections)

        # =================================================
        # DETECTION SUMMARY
        # =================================================

        st.divider()

        st.markdown("### 📈 Detection Summary")

        total_detections = len(df)

        unique_classes = (
            df["Fruit"].nunique()
            if not df.empty else 0
        )

        avg_confidence = (
            df["Confidence"].mean()
            if not df.empty else 0
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Detections", total_detections)

        with col2:
            st.metric("Fruit Categories", unique_classes)

        with col3:
            st.metric("Average Confidence", f"{avg_confidence:.2f}%")

        with col4:
            st.metric("Processing Time", f"{processing_time:.3f}s")

        # =================================================
        # DETECTION DETAILS
        # =================================================

        st.divider()

        st.markdown("### 🔍 Detection Details")

        if not df.empty:

            tab1, tab2, tab3 = st.tabs([
                "Detection Table",
                "Class Distribution",
                "Download Results"
            ])

            with tab1:

                display_df = df.copy()

                display_df.index = range(1, len(display_df) + 1)

                st.dataframe(
                    display_df,
                    use_container_width=True
                )

            with tab2:

                class_counts = df["Fruit"].value_counts()

                st.bar_chart(class_counts)

                st.markdown("#### Detected Fruit Counts")

                for fruit, count in class_counts.items():

                    st.write(f"**{fruit}:** {count}")

            with tab3:

                # CSV download
                csv_data = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Detection CSV",
                    data=csv_data,
                    file_name="fruit_detection_results.csv",
                    mime="text/csv"
                )

                # Annotated image download
                output_image = Image.fromarray(annotated_image)

                image_buffer = io.BytesIO()

                output_image.save(image_buffer, format="PNG")

                st.download_button(
                    label="⬇️ Download Annotated Image",
                    data=image_buffer.getvalue(),
                    file_name="fruit_detection_result.png",
                    mime="image/png"
                )

        else:

            st.warning(
                "No fruits detected. Try lowering the confidence threshold."
            )

    except Exception as e:

        st.error("An error occurred during image processing.")

        st.exception(e)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#8993a4;">
        <p>Fruit Detection System | Powered by YOLOv8</p>
        <p>Custom-trained model with Apple, Avocado, Banana, Mango and Orange classes.</p>
    </div>
    """,
    unsafe_allow_html=True
)
