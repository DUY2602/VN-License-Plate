from ultralytics import YOLO
import easyocr
import cv2

# Load model
model = YOLO("runs/detect/outputs/license_plate_v1/weights/best.pt")
reader = easyocr.Reader(['en'])

def predict(image_path):
    results = model(image_path)
    img = cv2.imread(image_path)

    plates = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box on image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"conf: {conf:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # OCR
            cropped = img[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2)
            gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            text = reader.readtext(gray, detail=0, allowlist='0123456789ABCDEFGHKLMNPSTUVXYZ-')
            plates.append("".join(text))

    # Save result image
    cv2.imwrite("result.jpg", img)
    print("Saved result.jpg")
    return plates

print(predict("test.jpg"))