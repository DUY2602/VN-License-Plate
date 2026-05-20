from ultralytics import YOLO

model = YOLO("model/yolov8n.pt")

model.train(
    data="data/data.yaml",
    epochs=10,
    batch=8,
    imgsz=416,
    save=True,
    save_period=10,
    exist_ok=True,
)