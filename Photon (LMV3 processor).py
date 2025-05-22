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
from tabulate import tabulate # Import tabulate for better table formatting (optional)

# Load the processor and model
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the device (CPU or GPU)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5).to(device)

# Define the label mapping
id2label = {
    0: "O",  # Outside any entity
    1: "Restaurant Name",
    2: "Item Name",
    3: "Item Qty",
    4: "Item Rate"
}

# Suppress transformers logging
logging.set_verbosity_error()

# Load the image using OpenCV
image_raw = cv2.imread("./bill.jpg")  # Load the image

# Load and pre-process the image using OpenCV
image_raw = cv2.imread("./bill.jpg")  # Load the image
gray_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY) # 1. Convert to Grayscale
_, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY) # 2. Apply Thresholding (Binarization)
resized_image = cv2.resize(binary_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) # 3. Resize the Image (Optional, if text is too small)
denoised_image = cv2.medianBlur(resized_image, 3) # 4. Noise Removal (Optional, if the image has noise)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 5. Apply Morphological Closing to Close Gaps
closed_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel) # 5. Apply Morphological Closing to Close Gaps

# Save the preprocessed image for Tesseract
preprocessed_image_path = "./preprocessed_image.jpg"
cv2.imwrite(preprocessed_image_path, closed_image)

image = closed_image  # Use the preprocessed image for OCR
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # Convert back to RGB for Tesseract

# Encode the image and bounding boxes using the processor
encoded_inputs = processor(image, return_tensors="pt", truncation=True)

print("Encoded inputs keys:", encoded_inputs.keys())
print("Encoded inputs shapes:", {k: v.shape for k, v in encoded_inputs.items()})

# Extract the first 10 items from encoded_inputs
input_ids = encoded_inputs["input_ids"][0][60:100].tolist()  # Token IDs
attention_mask = encoded_inputs["attention_mask"][0][60:100].tolist()  # Attention mask
bounding_boxes_normalized = encoded_inputs["bbox"][0][60:100].tolist()  # Bounding boxes

# Decode the token IDs back to text
decoded_tokens = [processor.decode([token_id]).strip() for token_id in input_ids]

# # Create a table with the data
# table_data = []
# for i in range(len(input_ids)):
#     table_data.append({
#         "Token ID": input_ids[i],
#         "Text": decoded_tokens[i],
#         "Attention Mask": attention_mask[i],
#         "Bounding Box": bounding_boxes_normalized[i]
#     })

# print(encoded_inputs.keys())
# print(tabulate(table_data, headers="keys", tablefmt="grid"))

# Move the encoded inputs to the same device as the model
encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()}

with torch.no_grad():
    outputs = model(**encoded_inputs)

predicted_class_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()

predicted_labels = [id2label[class_id] for class_id in predicted_class_ids]

classified_data = []
for token, bbox, label in zip(decoded_tokens, bounding_boxes_normalized, predicted_labels):
    classified_data.append({
        "Token": token,
        "Bounding Box": bbox,
        "Label": label
    })

from tabulate import tabulate
print(tabulate(classified_data, headers="keys", tablefmt="grid"))

# Get the original image dimensions (before resizing for display)
orig_height, orig_width = closed_image.shape[:2]

#-----------------------------------------------------------------------------------

# Unnormalize bounding boxes from [0, 1000] to pixel coordinates
unnormalized_bboxes = [
    [
        int(bbox[0] / 1000 * orig_width),
        int(bbox[1] / 1000 * orig_height),
        int(bbox[2] / 1000 * orig_width),
        int(bbox[3] / 1000 * orig_height)
    ]
    for bbox in bounding_boxes_normalized
]

# Highlight the bounding boxes for the first 10 items
for bbox in unnormalized_bboxes:
    # Draw a rectangle on the image
    cv2.rectangle(
        closed_image,
        (bbox[0], bbox[1]),  # Top-left corner
        (bbox[2], bbox[3]),  # Bottom-right corner
        color=(0, 255, 0),  # Green color
        thickness=1  # Thickness of the rectangle
    )

# Resize the image to fit the screen
window_width = 400  # Desired width of the window
window_height = 700  # Desired height of the window
closed_image = cv2.resize(closed_image, (window_width, window_height), interpolation=cv2.INTER_AREA)

cv2.imshow("Document Image", closed_image)  # Display the image in a new window
# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()
# #-----------------------------------------------------------------------------------