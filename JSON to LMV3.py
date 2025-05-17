import json
import os
from PIL import Image

LABEL_MAP = {
    "O": 0,
    "Restaurant Name": 1,
    "Item Name": 2,
    "Item Qty": 3,
    "Item Rate": 4,
}

def convert_labelstudio_to_layoutlm(labelstudio_json, image_dir, output_path):
    with open(labelstudio_json, "r") as f:
        data = json.load(f)
    layoutlm_samples = []
    for task in data:
        image_path = os.path.join(image_dir, task["file_upload"])
        for ann in task["annotations"]:
            words, boxes, labels = [], [], []
            for result in ann["result"]:
                if result["type"] == "textarea":
                    text = result["value"]["text"][0]
                    label = result["from_name"]
                    bbox = next(r["value"] for r in ann["result"] if r["id"] == result["id"] and r["type"] == "rectangle")
                    x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
                    # Convert to absolute coordinates
                    with Image.open(image_path) as img:
                        width, height = img.size
                    left = int(x / 100 * width)
                    top = int(y / 100 * height)
                    right = int((x + w) / 100 * width)
                    bottom = int((y + h) / 100 * height)
                    words.append(text)
                    boxes.append([left, top, right, bottom])
                    labels.append(LABEL_MAP.get(label, 0))
            layoutlm_samples.append({
                "image_path": image_path,
                "words": words,
                "boxes": boxes,
                "labels": labels
            })
    with open(output_path, "w") as f:
        json.dump(layoutlm_samples, f, indent=2)

#Usage:
convert_labelstudio_to_layoutlm("project-1-at-2025-05-17-11-03-bcab81a3.json", ".\Dataset", "layoutlm_train.json")