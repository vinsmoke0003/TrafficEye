import cv2
import numpy as np
import easyocr
from pathlib import Path

def initialize_easyocr():
    return easyocr.Reader(['en'])

def detect_plate_region(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.03 * cv2.arcLength(cnt, True), True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        area = cv2.contourArea(cnt)

        if 2.5 < aspect_ratio < 6 and area > 1000 and y > 200 and h > 20 and w > 80:
            return image[y:y+h, x:x+w], (x, y, w, h)
    
    return None, None

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def extract_text_easyocr(reader, plate_img):
    rgb_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_img, batch_size=1, text_threshold=0.5)
    if results:
        best_result = max(results, key=lambda x: x[2])
        return best_result[1].upper().replace(" ", "")
    return ""

def save_image(img, path):
    cv2.imwrite(str(path), img)
    print(f"Saved: {path}")

if __name__ == "__main__":
    input_path = Path("C:/Users/seenu/OneDrive/Desktop/programming/Python/project/data/sample.jpg")
    output_dir = Path("C:/Users/seenu/OneDrive/Desktop/programming/Python/project/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = initialize_easyocr()
    image = cv2.imread(str(input_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at: {input_path}")
    image = cv2.resize(image, (800, 600))

    plate_img, coords = detect_plate_region(image)
    if plate_img is None:
        print("❌ No license plate-like region detected.")
    else:
        processed_plate = preprocess_plate(plate_img)
        text = extract_text_easyocr(reader, processed_plate)

        print(f"✅ Detected License Plate: {text if text else 'NOT FOUND'}")

        save_image(image, output_dir / "original.jpg")
        save_image(plate_img, output_dir / "plate_extracted.jpg")
        save_image(processed_plate, output_dir / "plate_processed.jpg")
