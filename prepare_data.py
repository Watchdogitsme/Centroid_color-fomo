import os
import json
import csv
from glob import glob

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
CSV_OUTPUT = os.path.join(DATA_DIR, "annotations.csv")

CLASS_MAP = {
    "background": 0,
    "red": 1,
    "green": 2,
    "blue": 3,
    "yellow": 4
}

def parse_edge_impulse():
    records = []
    
    for split in ["training", "testing"]:
        split_dir = os.path.join(RAW_DIR, split)
        label_file = os.path.join(split_dir, "bounding_boxes.labels")
        
        if not os.path.exists(label_file):
            continue
            
        with open(label_file, "r") as f:
            data = json.load(f)
            
        boxes_dict = data.get("boundingBoxes", {})
        
        for filename, boxes in boxes_dict.items():
            img_path = os.path.join(split_dir, filename)
            
            # Check if file exists directly or fuzzy match
            if not os.path.exists(img_path):
                matches = glob(os.path.join(split_dir, f"*{filename}*"))
                if matches:
                    img_path = matches[0]
                else:
                    continue

            for bb in boxes:
                label_str = str(bb.get("label", "")).strip().lower()
                label_id = CLASS_MAP.get(label_str, 0)
                
                if label_id == 0:
                    continue
                    
                x = int(bb["x"])
                y = int(bb["y"])
                w = int(bb.get("width", bb.get("w", 0)))
                h = int(bb.get("height", bb.get("h", 0)))
                
                records.append([img_path, x, y, x + w, y + h, label_id])

    os.makedirs(DATA_DIR, exist_ok=True)
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "x_min", "y_min", "x_max", "y_max", "label_id"])
        writer.writerows(records)

    print(f"Extraction complete! {len(records)} bounding boxes saved to {CSV_OUTPUT}")

if __name__ == "__main__":
    parse_edge_impulse()