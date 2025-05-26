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

# Load the tokenizer
tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Specify the device (CPU or GPU)
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5).to(device)

# Suppress transformers logging
logging.set_verbosity_error()

# Define the label mapping
id2label = {
    0: "O",  # Outside any entity
    1: "Restaurant Name",
    2: "Item Name",
    3: "Item Qty",
    4: "Item Rate"
}

def preprocess_image(image_path):       # Load and pre-process the image using OpenCV
    image_raw = cv2.imread(image_path)  # Load the image
    gray_image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY) # 1. Convert to Grayscale
    _, binary_image = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY) # 2. Apply Thresholding (Binarization)
    resized_image = cv2.resize(binary_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR) # 3. Resize the Image (Optional, if text is too small)
    denoised_image = cv2.medianBlur(resized_image, 3) # 4. Noise Removal (Optional, if the image has noise)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 5. Apply Morphological Closing to Close Gaps
    closed_image = cv2.morphologyEx(denoised_image, cv2.MORPH_CLOSE, kernel) # 5. Apply Morphological Closing to Close Gaps
    return closed_image

# image = preprocess_image("./bill.jpg")  # Use the preprocessed image for OCR

def extract_ocr_data(image):        # Extract text and bounding boxes using Tesseract OCR
    ocr_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    texts = ocr_data["text"]  # Extract words and their bounding boxes
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
    return texts, bounding_boxes

#texts, bounding_boxes = extract_ocr_data(image)

def normalize_bboxes(bounding_boxes, image_shape):# Normalize bounding boxes to the range 0-1000
    image_width, image_height = image_shape[1], image_shape[0]  # Get image dimensions
    return [
        [
            int((bbox[0] / image_width) * 1000),  # Normalize left
            int((bbox[1] / image_height) * 1000),  # Normalize top
            int((bbox[2] / image_width) * 1000),  # Normalize right
            int((bbox[3] / image_height) * 1000)  # Normalize bottom
        ]
        for bbox in bounding_boxes
    ]

# bounding_boxes_normalized = normalize_bboxes(bounding_boxes, image.shape)

def classify_tokens(texts, bounding_boxes_normalized, tokenizer, model, device):
    # Tokenize the OCR output
    encoded_inputs = tokenizer(
        text=texts,
        boxes=bounding_boxes_normalized,
        return_tensors="pt",
        truncation=True
    )
    encoded_inputs = {key: value.to(device) for key, value in encoded_inputs.items()} # Move the encoded inputs to the same device as the model
    with torch.no_grad():
        outputs = model(**encoded_inputs)
    predicted_class_ids = outputs.logits.argmax(dim=-1).squeeze().tolist()
    predicted_labels = [id2label[class_id] for class_id in predicted_class_ids]
    input_ids = encoded_inputs["input_ids"][0][:].tolist()  # Token IDs
    bounding_boxes_normalized = encoded_inputs["bbox"][0][:].tolist()  # Bounding boxes
    decoded_tokens = [tokenizer.decode([token_id]).strip() for token_id in input_ids] # Decode the token IDs back to text
    classified_data = []
    for token, bbox, label in zip(decoded_tokens, bounding_boxes_normalized, predicted_labels):
        classified_data.append({
            "Token": token,
            "Bounding Box": bbox,
            "Label": label
        })
    return classified_data

def process_image(image_path, tokenizer, model, device):
    image = preprocess_image(image_path)
    texts, bounding_boxes = extract_ocr_data(image)
    bounding_boxes_normalized = normalize_bboxes(bounding_boxes, image.shape)
    classified_data = classify_tokens(texts, bounding_boxes_normalized, tokenizer, model, device)
    return classified_data

if __name__ == "__main__":
    # Load model and tokenizer once
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5).to(device)

    # List of images to process
    image_paths = [
        "./Dataset/train 1.jpg",
        "./Dataset/train 2.jpg",
        "./Dataset/trian 3.jpg",
        "./Dataset/train 4.jpg",
        "./Dataset/train 5.jpg"
    ]

    for image_path in image_paths:
        classified_data = process_image(image_path, tokenizer, model, device)
        print(tabulate(classified_data, headers="keys", tablefmt="grid"))

# #-----------------------------------------------------------------------------------
# def visualize_bboxes(image, bounding_boxes):
#     for bbox in bounding_boxes: # Draw bounding boxes on the image
#         cv2.rectangle(          # Draw a rectangle on the image
#             image,
#             (bbox[0], bbox[1]),  # Top-left corner
#             (bbox[2], bbox[3]),  # Bottom-right corner
#             color=(0, 255, 0),  # Green color
#             thickness=1  # Thickness of the rectangle
#         )
#     window_width = 400  # Desired width of the window
#     window_height = 700  # Desired height of the window
#     image = cv2.resize(image, (window_width, window_height), interpolation=cv2.INTER_AREA)
#     return image

# cv2.imshow("Document Image", visualize_bboxes(image,bounding_boxes))  # Display the image in a new window
# # Wait for a key press to close the image window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# #-----------------------------------------------------------------------------------