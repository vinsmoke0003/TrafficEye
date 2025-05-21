import cv2
import numpy as np

image_path = "C:/Users/seenu/OneDrive/Desktop/programming/Python/project/data/sample.jpg"  # Adjust if needed
image = cv2.imread(image_path)

if image is None:
    print("Image not found.")
    exit()

image = cv2.resize(image, (800, 600))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 100, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_with_contours = image.copy()
cv2.drawContours(image_with_contours, contours, -1, (0, 255, 0), 2)
cv2.imshow("Original Image", image)
cv2.imshow("Edges", edges)
cv2.imshow("Contours", image_with_contours)

cv2.waitKey(0)
cv2.destroyAllWindows()