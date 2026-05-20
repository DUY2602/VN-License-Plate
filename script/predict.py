import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from script.ocr import read_plate
from ultralytics import YOLO
import cv2

model = YOLO("model/best.pt")

def predict(image_path, save_result=True, debug_ocr=False):
    results = model(image_path)
    img     = cv2.imread(image_path)

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf            = float(box.conf[0])
            cropped         = img[y1:y2, x1:x2]
            text            = read_plate(cropped, debug=debug_ocr)
            plates.append(text)

            # Vẽ bounding box + text lên ảnh
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{text} ({conf:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    if save_result:
        out_path = image_path.replace("samples/", "samples/result_")
        cv2.imwrite(out_path, img)
        print(f"Saved: {out_path}")

    return plates

if __name__ == "__main__":
    results = predict("samples/test.jpg", debug_ocr=True)
    print("Detected plates:", results)