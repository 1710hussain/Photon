#Preprocessing the image for OCR using OpenCV and Tesseract
import cv2
import numpy as np

# Apply Otsu's thresholding
image = cv2.imread('bill.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
# image = cv2.GaussianBlur(image, (5, 5), 0)

# # Apply Otsu's thresholding
# image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# image = image[1]  # Get the thresholded image from the tuple

# # Apply morphological operations
# kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel
# image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Closing operation

# if image is None:
#     print("Error: Image not found or unable to load.")
# else:
#     # Display the image
#     cv2.imshow('Processed Image', image)
#     cv2.waitKey(0)  # Wait for a key press to close the window
#     cv2.destroyAllWindows()

#Extracting text from the image using Tesseract OCR
import pytesseract
from pytesseract import Output
data = pytesseract.image_to_data(image, output_type=Output.DICT)
# print(data.keys())  # Print the keys of the dictionary to see what data is available
# print("Number of boxes:", len(data['text']))  # Number of detected text boxes
# print("Text:", data['text'])  # Print the detected text
# print("Confidences:", data['conf'])  # Print the confidence scores for each box 
# Define the bounding box category (e.g., width and height range)
min_width, max_width = 50, 200  # Example width range
min_height, max_height = 10, 50  # Example height range

print("Filtered Text within Specific Bounding Box Category:")
for i in range(len(data['text'])):
    if data['text'][i].strip():  # Only process non-empty text
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        if min_width <= w <= max_width and min_height <= h <= max_height:  # Check bounding box size
            text = data['text'][i]
            print(f"Text: '{text}' | Bounding Box: (x: {x}, y: {y}, w: {w}, h: {h})")