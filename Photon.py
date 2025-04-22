import cv2
import numpy as np

# Apply Otsu's thresholding
image = cv2.imread('bill.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Gaussian blur to reduce noise
image = cv2.GaussianBlur(image, (5, 5), 0)

# Apply Otsu's thresholding
image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
image = image[1]  # Get the thresholded image from the tuple

# Apply morphological operations
kernel = np.ones((3, 3), np.uint8)  # Define a 3x3 kernel
image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # Closing operation

if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Display the image
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()