import sys
import time
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2

# 1. Self-contained Centroid FOMO Architecture
class FOMOModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        features = mobilenet_v2(weights=None).features
        self.backbone = nn.Sequential(*features[:7])  # 8x downsampled feature map
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes + 1, kernel_size=1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# 2. Class Labels & BGR Visualization Colors
CLASS_NAMES = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow"}
CLASS_COLORS = {
    1: (0, 0, 255),    # Red
    2: (0, 255, 0),    # Green
    3: (255, 0, 0),    # Blue
    4: (0, 255, 255)   # Yellow
}

def run_live_stream(source=0, weight_path="fomo_cubes_finetuned.pth", conf_thresh=0.5):
    """
    source:
      - 0 or 2: Local camera index
      - "video.mp4": Local video path
      - "http://<IP>:<PORT>/video": Phone/IP camera stream
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FOMOModel(num_classes=4).to(device)
    
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded fine-tuned checkpoint: {weight_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint '{weight_path}' not found. Run python finetune.py first.")
        return

    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # OS-specific capture backend setup
    if isinstance(source, int):
        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            # Force MJPEG to prevent USB/IP timeout bottlenecks over usbipd
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    else:
        cap = cv2.VideoCapture(source)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Unable to open video source: '{source}'.")
        return

    print("Live stream active. Press 'q' in the display window to exit.")
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame unavailable.")
            break

        h, w = frame.shape[:2]

        # Preprocessing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0).to(device)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_classes = np.argmax(probs, axis=0)  # (12, 12)
        pred_confs = np.max(probs, axis=0)
        grid_h, grid_w = pred_classes.shape

        # Decode detected centroids
        for gy in range(grid_h):
            for gx in range(grid_w):
                cls_id = pred_classes[gy, gx]
                conf = pred_confs[gy, gx]

                if cls_id > 0 and conf >= conf_thresh:
                    cx = int((gx + 0.5) * (w / grid_w))
                    cy = int((gy + 0.5) * (h / grid_h))

                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    cv2.circle(frame, (cx, cy), 6, color, -1)
                    cv2.putText(
                        frame,
                        f"{CLASS_NAMES.get(cls_id, 'Cube')} {conf:.2f}",
                        (cx - 25, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

        # Framerate Calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Centroid FOMO (Fine-Tuned)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # If using WSL via usbipd, source=0 or source=2 (whichever RGB node is active)
    # For a video file test, change to: run_live_stream(source="test.mp4")
    run_live_stream(source=0)