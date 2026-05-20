# 🚗 Vietnamese License Plate Detection

An end-to-end license plate detection system for Vietnamese vehicles using YOLOv8. The model detects both car and motorcycle license plates in real-world conditions with **99.2% mAP50** accuracy.

## 📌 Project Overview

| | |
|---|---|
| **Model** | YOLOv8n (Nano) |
| **Task** | Object Detection |
| **Classes** | Car plate, Motorcycle plate |
| **Dataset** | 1,233 images / 1,401 instances |
| **mAP50** | 0.992 |
| **mAP50-95** | 0.866 |
| **Precision** | 0.988 |
| **Recall** | 0.979 |

## 🗂️ Project Structure

```
license-plate/
├── model/
│   └── best.pt           # Trained model weights
├── data/                 # Dataset (after download)
├── .env                  # API keys (not committed)
├── data_reader.py   # Download dataset from Roboflow
├── train.py              # Train YOLOv8 model
├── predict.py            # Run inference on images
└── requirements.txt      # Dependencies
```

## ⚙️ Requirements

- Python 3.10+
- pip

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
ultralytics
roboflow
python-dotenv
easyocr
opencv-python
```

---

## 🚀 Train From Scratch

Follow these steps to reproduce the training pipeline from scratch.

### Step 1 — Get Roboflow API Key

1. Sign up at [roboflow.com](https://roboflow.com) (free)
2. Go to **Settings → API**
3. Copy your API key

### Step 2 — Set Up Environment

Create a `.env` file in the project root:

```
API_KEY=your_roboflow_api_key_here
```

### Step 3 — Download Dataset

```bash
python data_reader.py
```

This downloads the [Vietnam License Plate dataset](https://universe.roboflow.com/vietnam-license/vietnam-license-plate-hjswj) from Roboflow with the following structure:

```
Vietnam-License-Plate-2/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

`data_reader.py`:
```python
import os
from roboflow import Roboflow
from dotenv import load_dotenv

load_dotenv()

rf = Roboflow(api_key=os.getenv("API_KEY"))
project = rf.workspace("vietnam-license").project("vietnam-license-plate-hjswj")
version = project.version(2)
dataset = version.download("yolov8")
```

### Step 4 — Train the Model

```bash
python train.py
```

`train.py`:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # downloads pretrained weights automatically

model.train(
    data="Vietnam-License-Plate-2/data.yaml",
    epochs=10,
    batch=8,
    imgsz=416,
    device="cpu",       # change to 0 if you have an NVIDIA GPU
    save=True,
    save_period=10,
    project="checkpoints",
    name="v1",
    exist_ok=True,
)
```

> 💡 **Tip:** Training on CPU (i7 gen 13) takes ~15 minutes for 10 epochs at imgsz=416.
> For faster training, use Google Colab (free T4 GPU) — reduces training time to ~3 minutes.

**Training results after 10 epochs:**

| Class | Precision | Recall | mAP50 | mAP50-95 |
|-------|-----------|--------|-------|----------|
| All | 0.988 | 0.979 | 0.992 | 0.866 |
| Car plate | 0.995 | 0.984 | 0.995 | 0.861 |
| Motorcycle plate | 0.981 | 0.973 | 0.989 | 0.871 |

Trained weights are saved to `checkpoints/v1/weights/best.pt`.

### Step 5 — Run Inference

```bash
python predict.py
```

`predict.py`:
```python
from ultralytics import YOLO
import easyocr
import cv2

model = YOLO("checkpoints/v1/weights/best.pt")
reader = easyocr.Reader(['en'])

def predict(image_path):
    results = model(image_path)
    img = cv2.imread(image_path)

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # Draw bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"conf: {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Crop and preprocess for OCR
            cropped = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            text = reader.readtext(
                gray,
                detail=0,
                allowlist='0123456789ABCDEFGHKLMNPSTUVXYZ-'
            )
            plates.append("".join(text))

    cv2.imwrite("result.jpg", img)
    print(f"Detected plates: {plates}")
    return plates

predict("test_car.jpg")
```

---

## 📊 Model Performance

The model was evaluated on 1,233 validation images containing 1,401 license plate instances.

- **mAP50 = 0.992** — detects 99.2% of license plates correctly
- **Precision = 0.988** — 98.8% of detections are true positives
- **Recall = 0.979** — finds 97.9% of all license plates in images

### Known Limitations

The model performs well under the following conditions:

✅ Daytime images with good lighting  
✅ Frontal or slightly angled shots (< 30°)  
✅ Clean, unobstructed license plates  

Performance degrades in:

❌ Low-light or nighttime conditions  
❌ Heavily occluded or dirty plates  
❌ Extreme viewing angles  

---

## 🛠️ Tech Stack

- [YOLOv8](https://github.com/ultralytics/ultralytics) — object detection
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — optical character recognition
- [OpenCV](https://opencv.org/) — image preprocessing
- [Roboflow](https://roboflow.com/) — dataset management

## 📄 Dataset

- **Source:** [Roboflow Universe — Vietnam License Plate](https://universe.roboflow.com/vietnam-license/vietnam-license-plate-hjswj/dataset/2)
- **License:** CC BY 4.0
- **Classes:** 2 (car plate, motorcycle plate)
- **Train / Val / Test split:** included in download