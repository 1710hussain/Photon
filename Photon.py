# Import necessary libraries
import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification, logging
import torch  # Ensure PyTorch is imported

# Suppress transformers logging
logging.set_verbosity_error()

# Load model and processor
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")

from PIL import Image

image_path = "./bill.jpg"
image = Image.open(image_path).convert("RGB")

encoded_inputs = processor(image, return_tensors="pt")

print("Encoded inputs keys:", encoded_inputs.keys())
print("Encoded inputs shapes:", {k: v.shape for k, v in encoded_inputs.items()})

with torch.no_grad():
    outputs = model(**encoded_inputs)

import torch
probabilities = torch.softmax(torch.tensor(outputs.logits), dim=1)
print("Probabilities shape:", probabilities.shape)