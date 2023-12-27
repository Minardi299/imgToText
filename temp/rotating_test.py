import cv2
import pytesseract
import numpy as np
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\Tesseract-OCR\\tesseract.exe'

# Read the image
image = cv2.imread('IMG_20231219_183628.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection to find edges
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Apply Hough Line Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Calculate rotation angle
angle = 0.0
if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        if a != 0:
            angle = np.degrees(theta)
            break

# Rotate the image
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

# Perform OCR on the rotated image
text = pytesseract.image_to_string(rotated_image)

# Print the detected text
print("Detected Text:")
print(text)

# Show the rotated image
cv2.imshow("Rotated Image", rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
