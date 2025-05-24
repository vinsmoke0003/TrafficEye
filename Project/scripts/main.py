from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

import sys
sys.path.append("C:/Users/seenu/OneDrive/Desktop/programming/Python/Project/scripts/extract_plate_region")

from extract_plate_region import detect_plate_region

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Video input path
video_path = "C:/Users/seenu/OneDrive/Desktop/programming/Python/object_detector/test1.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / (fps * 1.25))

# Folder to save plates
output_dir = Path("../Project/outputs/plates")
output_dir.mkdir(parents=True, exist_ok=True)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    results = model.track(source=frame, persist=True)
    detections = results[0].boxes

    if detections is not None:
        for i, box in enumerate(detections.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            vehicle_crop = frame[y1:y2, x1:x2]

            # Try to find license plate inside the cropped vehicle
            plate_img, _ = detect_plate_region(vehicle_crop)
            if plate_img is not None:
                save_path = output_dir / f"plate_f{frame_idx}_v{i}.jpg"
                cv2.imwrite(str(save_path), plate_img)
                print(f"✅ Saved plate: {save_path}")

    # Show frame with YOLO predictions
    cv2.imshow("Vehicle Detection", results[0].plot())
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
