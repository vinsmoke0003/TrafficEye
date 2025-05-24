import cv2
import numpy as np

def detect_plate_region(image):
    """
    Detects a potential license plate region in the given image.
    Returns the cropped plate image and its bounding box coordinates.
    """
    image = cv2.resize(image, (800, 600))
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
            plate_crop = image[y:y+h, x:x+w]
            return plate_crop, (x, y, w, h)

    return None, None

# Load and test the function on your image
image_path = "C:/Users/seenu/OneDrive/Desktop/programming/Python/project/data/sample2.webp"
image = cv2.imread(image_path)
if image is None:
    print("Image not found.")
    exit()

plate_img, bbox = detect_plate_region(image)

if plate_img is not None and bbox is not None:
    x, y, w, h = bbox
    # Draw yellow rectangle on the image
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)  # (B, G, R) for Yellow
    cv2.imshow("Detected Plate Region (with Box)", image)
    cv2.imshow("Extracted Plate", plate_img)
else:
    print("No plate-like region found.")
    cv2.imshow("Original Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()
