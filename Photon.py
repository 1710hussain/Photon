# Import necessary libraries
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer, logging
import torch  # Ensure PyTorch is imported

# Suppress transformers logging
logging.set_verbosity_error()


from PIL import Image

image_path = "./bill.jpg"
image = Image.open(image_path).convert("RGB")


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
input_ids = encoded_inputs["input_ids"][0][120:140].tolist()  # Token IDs
attention_mask = encoded_inputs["attention_mask"][0][120:140].tolist()  # Attention mask
bounding_boxes = encoded_inputs["bbox"][0][120:140].tolist()  # Bounding boxes

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

# Print the table in a readable format
print(tabulate(table_data, headers="keys", tablefmt="grid"))


# Open the image using OpenCV
image_cv = cv2.imread(image_path)  # Load the image

# Highlight the bounding boxes for the first 10 items
for bbox in bounding_boxes:
    # Draw a rectangle on the image
    cv2.rectangle(
        image_cv,
        (bbox[0], bbox[1]),  # Top-left corner
        (bbox[2], bbox[3]),  # Bottom-right corner
        color=(0, 255, 0),  # Green color
        thickness=1  # Thickness of the rectangle
    )

cv2.imshow("Document Image", image_cv)  # Display the image in a new window
# Wait for a key press to close the image window
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Load model and processor
# processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
# model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

# encoded_inputs = processor(image, return_tensors="pt")

# print("Encoded inputs keys:", encoded_inputs.keys())
# print("Encoded inputs shapes:", {k: v.shape for k, v in encoded_inputs.items()})

# with torch.no_grad():
#     outputs = model(**encoded_inputs)

# import torch
# probabilities = torch.softmax(torch.tensor(outputs.logits), dim=1)
# predicted_class_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()  # Convert to a list
# print("Probabilities shape:", probabilities.shape)

# # Map class IDs to labels
# id2label = {
#     0: "O",  # Outside any entity
#     1: "ENTITY"  # Example label for class ID 1
# }

# predicted_labels = [id2label[class_id] for class_id in predicted_class_ids]


