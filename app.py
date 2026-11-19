
import streamlit as st
import pandas as pd
import numpy as np
import time
import io

from PIL import Image
from pathlib import Path
from ultralytics import YOLO


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="FruitVision AI | Object Detection",
    page_icon="🍎",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 2. CUSTOM CSS
# =========================================================

st.markdown("""
<style>

    /* Main application */
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #273449;
    }

    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }

    /* Main heading */
    .main-heading {
        font-size: 36px;
        font-weight: 800;
        color: #f8fafc;
        letter-spacing: -1px;
        margin-bottom: 5px;
    }

    .sub-heading {
        font-size: 15px;
        color: #94a3b8;
        margin-bottom: 25px;
    }

    /* Section heading */
    .section-heading {
        font-size: 21px;
        font-weight: 700;
        color: #f8fafc;
        margin-top: 15px;
        margin-bottom: 12px;
    }

    /* Dashboard cards */
    .info-card {
        background: linear-gradient(
            135deg,
            #172338,
            #111827
        );
        border: 1px solid #29384e;
        border-radius: 13px;
        padding: 18px;
        min-height: 115px;
    }

    .card-label {
        color: #94a3b8;
        font-size: 13px;
        margin-bottom: 10px;
    }

    .card-value {
        color: #f8fafc;
        font-size: 28px;
        font-weight: 800;
    }

    .card-description {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    /* Status labels */
    .status {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 700;
        background-color: #064e3b;
        color: #6ee7b7;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 12px;
        padding-top: 30px;
        border-top: 1px solid #29384e;
        margin-top: 40px;
    }

    /* Buttons */
    .stDownloadButton button {
        border-radius: 8px;
        font-weight: 600;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #111827;
        border: 1px dashed #475569;
        border-radius: 12px;
        padding: 15px;
    }

    /* Metrics */
    [data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #29384e;
        padding: 15px;
        border-radius: 12px;
    }

    [data-testid="stMetricLabel"] {
        color: #94a3b8;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        color: #94a3b8;
    }

</style>
""", unsafe_allow_html=True)


# =========================================================
# 3. MODEL CONFIGURATION
# =========================================================

MODEL_PATH = Path(__file__).parent / "weights" / "best.pt"

CLASS_NAMES = [
    "Apple",
    "Avocado",
    "Banana",
    "Mango",
    "Orange"
]


@st.cache_resource
def load_model(model_path):
    return YOLO(str(model_path))


# =========================================================
# 4. LOAD MODEL
# =========================================================

if not MODEL_PATH.exists():

    st.error(
        "Model file not found. Please place best.pt "
        "inside the weights folder."
    )

    st.stop()


try:
    model = load_model(str(MODEL_PATH))

except Exception as e:

    st.error(f"Model loading failed: {e}")
    st.stop()


# =========================================================
# 5. SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div style="text-align:center; padding:15px 0;">
            <div style="font-size:42px;">🍎</div>
            <div style="font-size:23px;font-weight:800;">
                FruitVision AI
            </div>
            <div style="font-size:12px;color:#94a3b8;">
                Intelligent Fruit Detection
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.05,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Minimum confidence required to display a detection."
    )

    iou_threshold = st.slider(
        "IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.70,
        step=0.05,
        help="Controls how overlapping bounding boxes are filtered."
    )

    image_size = st.selectbox(
        "Image Resolution",
        options=[320, 416, 512, 640],
        index=3
    )

    st.divider()

    st.markdown("### 🧠 Model Information")

    st.markdown(
        """
        **Architecture:** YOLOv8 Nano

        **Framework:** Ultralytics

        **Task:** Object Detection

        **Classes:** 5

        **Training Epochs:** 20
        """
    )

    st.divider()

    st.markdown("### 🍇 Supported Fruits")

    for fruit in CLASS_NAMES:
        st.markdown(f"• {fruit}")

    st.divider()

    st.caption("FruitVision AI | Academic Project")


# =========================================================
# 6. HEADER
# =========================================================

header_col1, header_col2 = st.columns([4, 1])

with header_col1:

    st.markdown(
        '<div class="main-heading">Fruit Detection Dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-heading">'
        'AI-powered object detection using a custom-trained YOLOv8 model.'
        '</div>',
        unsafe_allow_html=True
    )

with header_col2:

    st.markdown(
        '<div style="text-align:right;padding-top:10px;">'
        '<span class="status">● MODEL ONLINE</span>'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# 7. TOP OVERVIEW
# =========================================================

overview_cols = st.columns(3)

overview_data = [
    ("AI MODEL", "YOLOv8 Nano", "Custom trained object detector"),
    ("SUPPORTED CLASSES", "05", "Five different fruit categories"),
    ("TRAINING EPOCHS", "20", "Completed training cycles")
]

for col, (label, value, description) in zip(
    overview_cols,
    overview_data
):

    with col:

        st.markdown(
            f"""
            <div class="info-card">
                <div class="card-label">{label}</div>
                <div class="card-value">{value}</div>
                <div class="card-description">{description}</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# 8. IMAGE UPLOAD
# =========================================================

st.markdown(
    '<div class="section-heading">📤 Upload Image for Analysis</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a fruit image",
    type=["jpg", "jpeg", "png"],
    help="Upload an image containing one or more fruits."
)


# =========================================================
# 9. IMAGE PROCESSING
# =========================================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file).convert("RGB")

        image_array = np.array(image)

        # Image information
        st.caption(
            f"File: {uploaded_file.name} | "
            f"Dimensions: {image.width} × {image.height}"
        )

        # Run model
        with st.spinner("AI model is analyzing your image..."):

            start_time = time.perf_counter()

            results = model.predict(
                source=image_array,
                conf=confidence,
                iou=iou_threshold,
                imgsz=image_size,
                verbose=False
            )

            processing_time = (
                time.perf_counter() - start_time
            ) * 1000

        result = results[0]

        # Generate annotated image
        annotated_image = result.plot()
        annotated_image = annotated_image[:, :, ::-1]

        # =================================================
        # 10. DETECTION METRICS
        # =================================================

        detections = []

        for index, box in enumerate(result.boxes):

            class_id = int(box.cls[0].item())

            class_name = result.names[class_id]

            score = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({

                "Detection ID": index + 1,

                "Fruit": class_name,

                "Confidence (%)": round(score * 100, 2),

                "X1": round(x1, 2),

                "Y1": round(y1, 2),

                "X2": round(x2, 2),

                "Y2": round(y2, 2)

            })

        df = pd.DataFrame(detections)

        total_objects = len(detections)

        unique_classes = (
            df["Fruit"].nunique()
            if total_objects > 0 else 0
        )

        average_confidence = (
            df["Confidence (%)"].mean()
            if total_objects > 0 else 0
        )

        # =================================================
        # 11. KPI CARDS
        # =================================================

        st.markdown(
            '<div class="section-heading">📊 Analysis Overview</div>',
            unsafe_allow_html=True
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        metric1.metric(
            "Total Detections",
            total_objects
        )

        metric2.metric(
            "Unique Classes",
            unique_classes
        )

        metric3.metric(
            "Average Confidence",
            f"{average_confidence:.2f}%"
        )

        metric4.metric(
            "Processing Time",
            f"{processing_time:.2f} ms"
        )

        st.divider()

        # =================================================
        # 12. IMAGE COMPARISON
        # =================================================

        st.markdown(
            '<div class="section-heading">🖼️ Visual Detection Results</div>',
            unsafe_allow_html=True
        )

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            st.markdown("#### Original Image")

            st.image(
                image,
                use_container_width=True
            )

        with image_col2:

            st.markdown("#### AI Annotated Image")

            st.image(
                annotated_image,
                use_container_width=True
            )

        st.divider()

        # =================================================
        # 13. RESULTS TABS
        # =================================================

        tab1, tab2, tab3 = st.tabs([
            "📋 Detection Details",
            "📈 Class Distribution",
            "🧠 Model Information"
        ])

        # TAB 1: Detection details
        with tab1:

            st.subheader("Detected Objects")

            if total_objects > 0:

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                # Download CSV
                csv_data = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Detection Report (CSV)",
                    data=csv_data,
                    file_name="fruit_detection_report.csv",
                    mime="text/csv"
                )

            else:

                st.warning(
                    "No fruits were detected. "
                    "Try lowering the confidence threshold."
                )

        # TAB 2: Distribution
        with tab2:

            st.subheader("Fruit Class Distribution")

            if total_objects > 0:

                class_counts = (
                    df["Fruit"]
                    .value_counts()
                    .reindex(CLASS_NAMES, fill_value=0)
                )

                chart_data = pd.DataFrame({
                    "Fruit": class_counts.index,
                    "Detections": class_counts.values
                })

                st.bar_chart(
                    chart_data.set_index("Fruit")
                )

                st.dataframe(
                    chart_data,
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No class distribution is available.")

        # TAB 3: Model information
        with tab3:

            st.subheader("Model Configuration")

            model_information = pd.DataFrame({

                "Parameter": [
                    "Architecture",
                    "Model Variant",
                    "Task",
                    "Training Epochs",
                    "Number of Classes",
                    "Image Size",
                    "Confidence Threshold",
                    "IoU Threshold"
                ],

                "Value": [
                    "YOLOv8",
                    "Nano",
                    "Object Detection",
                    "20",
                    "5",
                    str(image_size),
                    str(confidence),
                    str(iou_threshold)
                ]

            })

            st.dataframe(
                model_information,
                use_container_width=True,
                hide_index=True
            )

        # =================================================
        # 14. DOWNLOAD ANNOTATED IMAGE
        # =================================================

        st.divider()

        st.markdown(
            '<div class="section-heading">⬇️ Export Results</div>',
            unsafe_allow_html=True
        )

        image_buffer = io.BytesIO()

        Image.fromarray(annotated_image).save(
            image_buffer,
            format="PNG"
        )

        st.download_button(
            label="Download Annotated Image",
            data=image_buffer.getvalue(),
            file_name="fruit_detection_result.png",
            mime="image/png"
        )

    except Exception as e:

        st.error(f"An error occurred during image processing: {e}")


else:

    # Empty dashboard state
    st.markdown(
        """
        <div style="
            background:#111827;
            border:1px dashed #475569;
            border-radius:15px;
            padding:45px;
            text-align:center;
            margin-top:20px;
        ">
            <div style="font-size:55px;">🍎</div>
            <h3 style="color:#f8fafc;">
                Ready to Analyze Your Image
            </h3>
            <p style="color:#94a3b8;">
                Upload an image above to start AI-powered
                fruit detection.
            </p>
            <p style="color:#64748b;font-size:13px;">
                Supported formats: JPG, JPEG, PNG
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">
        <p><b>FruitVision AI</b> | YOLOv8 Object Detection System</p>
        <p>Built with Python, Ultralytics YOLO and Streamlit</p>
        <p>Academic Computer Vision Project | 2026</p>
    </div>
    """,
    unsafe_allow_html=True
)
