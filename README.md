# 🍎 Fruit Detection Using YOLOv8

An AI-powered fruit detection application built using **YOLOv8 (Ultralytics)** and **Streamlit**. The system detects and classifies five different fruit categories in uploaded images and displays annotated detection results with confidence scores.

<p align="center">
  <a href="https://fruit-detection-using-yolov8-apxcptu9sqzjubfjsjhrgf.streamlit.app/">
    <strong>🚀 Try the Live Application</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/Khubaib-Muhammad/Fruit-Detection-using-YOLOv8">
    <strong>📂 GitHub Repository</strong>
  </a>
</p>

---

## 📌 Project Overview

This project implements an object detection system using a custom-trained YOLOv8 Nano model.

Users can upload an image containing fruits, adjust detection parameters, and receive predictions showing the detected fruit categories, bounding boxes, and confidence scores.

The application provides an interactive interface developed with Streamlit and uses the trained YOLOv8 model for real-time image inference.

## 🍏 Supported Fruit Classes

The model is trained to detect the following five fruit categories:

| Class ID | Fruit |
|---|---|
| 0 | Apple |
| 1 | Avocado |
| 2 | Banana |
| 3 | Mango |
| 4 | Orange |

## ✨ Features

- **Image Upload:** Upload JPG, JPEG, and PNG images.
- **Object Detection:** Detect multiple fruits in a single image.
- **Bounding Boxes:** Display detected fruits with annotated bounding boxes.
- **Confidence Scores:** View the confidence score for each detection.
- **Adjustable Confidence:** Control the minimum detection confidence threshold.
- **Adjustable IoU:** Configure the Intersection over Union threshold.
- **Image Resolution:** Select the inference image size.
- **Detection Summary:** View total detections, unique fruit categories, average confidence, and processing time.
- **Class Distribution:** Visualize the number of detected fruits per category.
- **Download Results:** Export detection information as CSV and download annotated images.

## 🧠 Model Information

| Parameter | Details |
|---|---|
| Model Architecture | YOLOv8 Nano |
| Framework | Ultralytics |
| Task | Object Detection |
| Number of Classes | 5 |
| Training Epochs | 20 |
| Default Image Size | 640 × 640 |
| Model Weights | `weights/best.pt` |
| Inference Device | CPU on Streamlit Cloud |

## 📊 Dataset Information

The custom fruit dataset contains five object classes.

### Training Distribution

| Fruit | Number of Instances |
|---|---:|
| Apple | 86 |
| Avocado | 179 |
| Banana | 97 |
| Mango | 128 |
| Orange | 233 |
| **Total** | **723** |

### Validation Distribution

| Fruit | Number of Instances |
|---|---:|
| Apple | 10 |
| Avocado | 20 |
| Banana | 17 |
| Mango | 10 |
| Orange | 32 |
| **Total** | **89** |

The dataset uses YOLO-format object detection annotations, with class IDs and normalized bounding-box coordinates.

## 📈 Model Evaluation

The trained model was evaluated on the validation dataset.

Latest recorded validation metrics:

| Metric | Score |
|---|---:|
| Precision | 96.3% |
| Recall | 93.4% |
| mAP@50 | 95.8% |
| mAP@50-95 | 72.8% |

**Note:** These are validation results, not a guarantee of performance on every external image. Results may vary depending on image quality, lighting, fruit appearance, and background conditions.

## 🛠️ Technologies Used

- **Python** – Main programming language
- **Ultralytics YOLOv8** – Object detection model
- **PyTorch** – Deep learning framework
- **OpenCV Headless** – Image processing dependencies
- **Pillow (PIL)** – Image loading and conversion
- **NumPy** – Numerical operations
- **Pandas** – Detection results and CSV generation
- **Streamlit** – Interactive web application
- **Google Colab** – Model training and experimentation
- **Git & GitHub** – Version control and project hosting

## 📂 Repository Structure

```text
Fruit-Detection-using-YOLOv8/
│
├── weights/
│   └── best.pt
│
├── YOLO_Fruit_Object_Detection.ipynb
│
├── app.py
│
├── requirements.txt
│
├── packages.txt
│
├── .gitignore
│
└── README.md
```

### File Description

| File / Folder | Description |
|---|---|
| `weights/best.pt` | Trained YOLOv8 model weights |
| `YOLO_Fruit_Object_Detection.ipynb` | Model training, validation, and evaluation notebook |
| `app.py` | Streamlit application |
| `requirements.txt` | Python dependencies |
| `packages.txt` | Linux system dependencies for deployment |
| `.gitignore` | Files excluded from Git tracking |
| `README.md` | Project documentation |

## ⚙️ Installation and Setup

Follow these steps to run the project locally.

### 1. Clone the Repository

```bash
git clone https://github.com/Khubaib-Muhammad/Fruit-Detection-using-YOLOv8.git
```

### 2. Navigate to the Project Directory

```bash
cd Fruit-Detection-using-YOLOv8
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser, usually at:

```text
http://localhost:8501
```

## 🚀 Live Deployment

The application is deployed using **Streamlit Community Cloud**.

🔗 **Live Application:**

https://fruit-detection-using-yolov8-apxcptu9sqzjubfjsjhrgf.streamlit.app/

🔗 **GitHub Repository:**

https://github.com/Khubaib-Muhammad/Fruit-Detection-using-YOLOv8

## 🔍 How to Use the Application

1. Open the live Streamlit application.
2. Upload an image containing one or more fruits.
3. Adjust the confidence threshold if needed.
4. Select the desired IoU threshold and inference resolution.
5. View the original image and YOLOv8 annotated image.
6. Explore the detection summary and class distribution.
7. Download the detection report or annotated image.

## 🔮 Future Improvements

Potential improvements include:

- Expanding the dataset with more fruit images and variations.
- Improving detection performance on unfamiliar images.
- Adding support for additional fruit categories.
- Implementing video-based fruit detection.
- Exploring real-time camera-based detection.
- Further optimizing inference speed and deployment performance.

## 👨‍💻 Author

**Khubaib Muhammad**

GitHub: [Khubaib-Muhammad](https://github.com/Khubaib-Muhammad)

---

<p align="center">
  Developed using Python, YOLOv8, PyTorch, and Streamlit.
</p>
