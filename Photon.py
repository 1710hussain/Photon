#Preprocessing the image for OCR using OpenCV and Tesseract
import cv2
import numpy as np

# Apply Otsu's thresholding
image = cv2.imread('bill.jpg', cv2.IMREAD_GRAYSCALE)

#Extracting text from the image using Tesseract OCR
import pytesseract
from pytesseract import Output
data = pytesseract.image_to_data(image, output_type=Output.DICT)

from transformers import LayoutLMTokenizer, logging

# Suppress transformers logging
logging.set_verbosity_error()

# Initialize LayoutLM tokenizer
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")
# Prepare text and bounding boxes
texts = []
bounding_boxes = []
image_width, image_height = image.shape[1], image.shape[0]  # Get image dimensions

for i in range(len(data['text'])):
    if data['text'][i].strip():  # Only process non-empty text
        texts.append(data['text'][i])
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # Normalize bounding box coordinates to fit LayoutLM format (0-1000)
        bounding_boxes.append([
            int(1000 * x / image_width),  # left
            int(1000 * y / image_height),  # top
            int(1000 * (x + w) / image_width),  # right
            int(1000 * (y + h) / image_height)  # bottom
        ])

# Tokenize the text
encoded_inputs = tokenizer(
    texts,
    boxes=bounding_boxes,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)

# Store the tokenized data in another variable if needed
tokenized_data = encoded_inputs

# Optionally, print or inspect the tokenized data
print(tokenized_data)