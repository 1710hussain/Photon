#Label-Studio User account Details
# Email: hussain@test.com
# Password: hussain6421

# Import necessary libraries
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer, logging
import torch  # Ensure PyTorch is imported

# Suppress transformers logging
logging.set_verbosity_error()


# Load the image using OpenCV
image_raw = cv2.imread("./bill.jpg")  # Load the image

# Preprocessing Steps
# 1. Convert to Grayscale
gray_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)

# 2. Apply Thresholding (Binarization)
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)

# 3. Resize the Image (Optional, if text is too small)
resized_image = cv2.resize(binary_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

# 4. Noise Removal (Optional, if the image has noise)
denoised_image = cv2.medianBlur(resized_image, 3)

# 5. Apply Morphological Closing to Close Gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # Define a 3x3 rectangular kernel
closed_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel)


# Save the preprocessed image for Tesseract
preprocessed_image_path = "./preprocessed_image.jpg"
cv2.imwrite(preprocessed_image_path, closed_image)

# # Load the image using OpenCV
# image = cv2.imread(preprocessed_image_path)  # Replace with your image path

image = closed_image  # Use the preprocessed image for OCR

# Extract text and bounding boxes using Tesseract OCR
ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)

# Extract words and their bounding boxes
texts = ocr_data["text"]  # List of words
bounding_boxes = [
    [ocr_data["left"][i], ocr_data["top"][i], ocr_data["left"][i] + ocr_data["width"][i], ocr_data["top"][i] + ocr_data["height"][i]]
    for i in range(len(ocr_data["text"]))
]

# Filter out empty tokens and their corresponding bounding boxes
texts, bounding_boxes = zip(
    *[(text, bbox) for text, bbox in zip(texts, bounding_boxes) if text.strip()]
)
texts = list(texts)
bounding_boxes = list(bounding_boxes)

print("Texts:", len(texts))
print("Bounding boxes:", len(bounding_boxes))

# Normalize bounding boxes to the range 0-1000
image_width, image_height = image.shape[1], image.shape[0]  # Get image dimensions
bounding_boxes = [
    [
        int((bbox[0] / image_width) * 1000),  # Normalize left
        int((bbox[1] / image_height) * 1000),  # Normalize top
        int((bbox[2] / image_width) * 1000),  # Normalize right
        int((bbox[3] / image_height) * 1000)  # Normalize bottom
    ]
    for bbox in bounding_boxes
]

# Load the tokenizer
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")

# Tokenize the OCR output
encoded_inputs = tokenizer(
    text=texts,
    boxes=bounding_boxes,
    return_tensors="pt",
    truncation=True
)

print("Encoded inputs keys:", encoded_inputs.keys())
print("Encoded inputs shapes:", {k: v.shape for k, v in encoded_inputs.items()})

# Import tabulate for better table formatting (optional)
from tabulate import tabulate

# Extract the first 10 items from encoded_inputs
input_ids = encoded_inputs["input_ids"][0][:].tolist()  # Token IDs
attention_mask = encoded_inputs["attention_mask"][0][:].tolist()  # Attention mask
bounding_boxes = encoded_inputs["bbox"][0][:].tolist()  # Bounding boxes

# Decode the token IDs back to text
decoded_tokens = [tokenizer.decode([token_id]).strip() for token_id in input_ids]

# Create a table with the data
table_data = []
for i in range(len(input_ids)):
    table_data.append({
        "Token ID": input_ids[i],
        "Text": decoded_tokens[i],
        "Attention Mask": attention_mask[i],
        "Bounding Box": bounding_boxes[i]
    })

# Specify the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5).to(device)

# Move the encoded inputs to the same device as the model
encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

with torch.no_grad():
    outputs = model(**encoded_inputs)

predicted_class_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()

id2label = {
    0: "O",  # Outside any entity
    1: "Restaurant Name",
    2: "GST No.",
    3: "Item Name",
    4: "Item Rate"
}

predicted_labels = [id2label[class_id] for class_id in predicted_class_ids]

classified_data = []
for token, bbox, label in zip(decoded_tokens, bounding_boxes, predicted_labels):
    classified_data.append({
        "Token": token,
        "Bounding Box": bbox,
        "Label": label
    })

from tabulate import tabulate
print(tabulate(classified_data, headers="keys", tablefmt="grid"))

#-----------------------------------------------------------------------------------
# # Highlight the bounding boxes for the first 10 items
# for bbox in bounding_boxes:
#     # Draw a rectangle on the image
#     cv2.rectangle(
#         closed_image,
#         (bbox[0], bbox[1]),  # Top-left corner
#         (bbox[2], bbox[3]),  # Bottom-right corner
#         color=(0, 255, 0),  # Green color
#         thickness=1  # Thickness of the rectangle
#     )

# # Resize the image to fit the screen
# window_width = 400  # Desired width of the window
# window_height = 700  # Desired height of the window
# closed_image = cv2.resize(closed_image, (window_width, window_height), interpolation=cv2.INTER_AREA)

# cv2.imshow("Document Image", closed_image)  # Display the image in a new window
# # Wait for a key press to close the image window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#-----------------------------------------------------------------------------------