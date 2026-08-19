import cv2
import torch
import numpy as np
import glob
from train import FOMOModel
from torchvision import transforms

# 1. Update dictionary mappings for all 4 classes
CLASS_COLORS = {
    1: (0, 0, 255),    # Red (BGR)
    2: (0, 255, 0),    # Green (BGR)
    3: (255, 0, 0),    # Blue (BGR)
    4: (0, 255, 255)   # Yellow (BGR)
}
CLASS_NAMES = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow"}

def detect(img_path, weight_path="fomo_cubes.pth", conf_thresh=0.5):
    device = torch.device("cpu")
    
    # 2. Set num_classes=4 (Matches the 5-channel checkpoint output)
    model = FOMOModel(num_classes=4)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print(f"Failed to load image: {img_path}")
        return

    h, w = orig_img.shape[:2]
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).numpy()  # Shape: (5, 12, 12)

    pred_classes = np.argmax(probs, axis=0)  # (12, 12)
    pred_confs = np.max(probs, axis=0)       # (12, 12)

    vis_img = orig_img.copy()
    grid_h, grid_w = pred_classes.shape

    for gy in range(grid_h):
        for gx in range(grid_w):
            cls_id = pred_classes[gy, gx]
            conf = pred_confs[gy, gx]

            if cls_id > 0 and conf >= conf_thresh:
                # Map 12x12 grid cell back to full pixel space
                cx = int((gx + 0.5) * (w / grid_w))
                cy = int((gy + 0.5) * (h / grid_h))
                
                color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                cv2.circle(vis_img, (cx, cy), 6, color, -1)
                cv2.putText(
                    vis_img, 
                    f"{CLASS_NAMES.get(cls_id, 'Obj')} {conf:.2f}", 
                    (cx - 20, cy - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.4, 
                    (255, 255, 255), 
                    1
                )

    cv2.imwrite("output_detection.png", vis_img)
    print("Detection complete! Saved to output_detection.png")

if __name__ == "__main__":
    sample_images = glob.glob("data/raw/**/*.jpg", recursive=True) + glob.glob("data/raw/**/*.png", recursive=True)
    if sample_images:
        detect(sample_images[0])
    else:
        print("No sample image found to run inference on.")