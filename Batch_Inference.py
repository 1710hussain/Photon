from Photon import process_image
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer, logging
import torch
from tabulate import tabulate
import cv2

# Suppress transformers logging
logging.set_verbosity_error()

if __name__ == "__main__":
    # Load model and tokenizer once
    tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=5).to(device)

    # List of images to process
    image_paths = [
        "./Dataset/train 1.jpg",
        "./Dataset/train 2.jpg",
        "./Dataset/train 4.jpg",
        "./Dataset/train 5.jpg"
    ]

    for image_path in image_paths:
        classified_data = process_image(image_path, tokenizer, model, device)
        print(f"\nResults for {image_path}:")
        print(tabulate(classified_data, headers="keys", tablefmt="grid"))