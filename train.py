from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(
    data="Vietnam-License-Plate-2/data.yaml",
    epochs=10,
    batch=8,
    imgsz=416,
    save=True,
    save_period=10,
    project="outputs",
    name="license_plate_v1",
    exist_ok=True,
)