import cv2
import numpy as np
from pathlib import Path
import easyocr

def initialize_easyocr():
    return easyocr.Reader(['en'])

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

def extract_text_easyocr(reader, plate_img):
    rgb_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(rgb_img,
                            batch_size=1,
                            text_threshold=0.5)
    if results:
        best_result = max(results, key=lambda x: x[2])
        return best_result[1].upper().replace(" ", "")
    return ""

def save_processed_image(img, output_path):
    cv2.imwrite(str(output_path), img)
    print(f"Saved image to: {output_path}")

if __name__ == "__main__":
    reader = initialize_easyocr()
    input_path = "C:/Users/seenu/OneDrive/Desktop/programming/Python/project/data/sample2.webp"
    
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Image not found at: {input_path}")
    
    plate_img = cv2.imread(input_path)
    if plate_img is None:
        raise ValueError("Invalid image file")
    
    processed_plate = preprocess_plate(plate_img)
    plate_text = extract_text_easyocr(reader, processed_plate)
    
    print(f"Detected License Plate: {plate_text if plate_text else 'NOT FOUND'}")
    
    # Save outputs without displaying
    Path("../outputs").mkdir(exist_ok=True)
    save_processed_image(plate_img, "../outputs/original.jpg")
    save_processed_image(processed_plate, "../outputs/processed.jpg")