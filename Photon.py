# Import necessary libraries
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from transformers import LayoutLMTokenizer, LayoutLMForTokenClassification, logging

# Suppress transformers logging
logging.set_verbosity_error()

# Step 1: Load and preprocess the image
image = cv2.imread('bill.jpg', cv2.IMREAD_GRAYSCALE)  # Load the receipt image in grayscale

# Step 2: Extract text and bounding boxes using Tesseract OCR
data = pytesseract.image_to_data(image, output_type=Output.DICT)


# Step 3: Initialize LayoutLM tokenizer
tokenizer = LayoutLMTokenizer.from_pretrained("microsoft/layoutlm-base-uncased")

# Prepare text and bounding boxes
texts = []
bounding_boxes = []
image_width, image_height = image.shape[1], image.shape[0]  # Get image dimensions
print("Image dimensions:", image_width, image_height)

print(range(len(data['text'])))

for i in range(len(data['text'])):
    if data['text'][i].strip():  # Only process non-empty text
        texts.append(data['text'][i])
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        # Normalize bounding box coordinates to fit LayoutLM format (0-1000)
        left = max(0, min(1000, int(1000 * x / image_width)))
        top = max(0, min(1000, int(1000 * y / image_height)))
        right = max(0, min(1000, int(1000 * (x + w) / image_width)))
        bottom = max(0, min(1000, int(1000 * (y + h) / image_height)))
        bounding_boxes.append([left, top, right, bottom])

# Step 4: Tokenize the text and bounding boxes
encoded_inputs = tokenizer(
    texts,
    boxes=bounding_boxes,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
    max_length=512
)


# Store the tokenized data
tokenized_data = encoded_inputs
print("Tokenized data:", len(tokenized_data['input_ids']), len(tokenized_data['token_type_ids']), len(tokenized_data['attention_mask']))
print("Tokenized data:", len(tokenized_data['input_ids'][0]), len(tokenized_data['token_type_ids'][0]), len(tokenized_data['attention_mask'][0]))

# # Step 5: Load the pre-trained LayoutLM model
# model = LayoutLMForTokenClassification.from_pretrained("microsoft/layoutlm-base-uncased")

# # Step 6: Perform inference
# outputs = model(
#     input_ids=tokenized_data['input_ids'],
#     attention_mask=tokenized_data['attention_mask'],
#     bbox=tokenized_data['token_type_ids']
# )

# # Get the raw predictions (logits)
# logits = outputs.logits

# # Convert logits to predicted class IDs
# predictions = logits.argmax(dim=-1)

# # Step 7: Define label mapping
# id2label = {
#     0: "O",  # Outside any entity
#     1: "B-RESTAURANT",  # Beginning of restaurant name
#     2: "I-RESTAURANT",  # Inside restaurant name
#     3: "B-ITEM",  # Beginning of item name
#     4: "I-ITEM",  # Inside item name
#     5: "B-QUANTITY",  # Beginning of quantity
#     6: "I-QUANTITY",  # Inside quantity
#     7: "B-RATE",  # Beginning of rate
#     8: "I-RATE",  # Inside rate
#     9: "B-TOTAL",  # Beginning of total
#     10: "I-TOTAL"  # Inside total
# }

# # Map predictions to labels
# predicted_labels = [id2label[label_id] for label_id in predictions[0].tolist()]

# # Step 8: Combine tokens, bounding boxes, and labels
# classified_data = []
# for token, bbox, label in zip(texts, bounding_boxes, predicted_labels):
#     if label != "O":  # Ignore tokens labeled as "O" (outside any entity)
#         classified_data.append({"text": token, "bbox": bbox, "label": label})

# # Step 9: Organize the classified data into sections
# restaurant_name = " ".join([item['text'] for item in classified_data if "RESTAURANT" in item['label']])
# items = [item['text'] for item in classified_data if "ITEM" in item['label']]
# quantities = [item['text'] for item in classified_data if "QUANTITY" in item['label']]
# rates = [item['text'] for item in classified_data if "RATE" in item['label']]
# totals = [item['text'] for item in classified_data if "TOTAL" in item['label']]

# # Step 10: Print the organized data
# print("Restaurant Name:", restaurant_name)
# print("Items:", items)
# print("Quantities:", quantities)
# print("Rates:", rates)
# print("Totals:", totals)