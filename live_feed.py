import cv2
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
from torchvision import transforms
import numpy as np
import time

# 1. Self-contained FOMO Model Architecture
class FOMOModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        features = mobilenet_v2(weights=None).features
        self.backbone = nn.Sequential(*features[:7])  # Downsampled 8x
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes + 1, kernel_size=1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# 2. Labels and Colors
CLASS_NAMES = {1: "Red", 2: "Green", 3: "Blue", 4: "Yellow"}
CLASS_COLORS = {
    1: (0, 0, 255),    # Red
    2: (0, 255, 0),    # Green
    3: (255, 0, 0),    # Blue
    4: (0, 255, 255)   # Yellow
}

def run_stream(source=0, weight_path="fomo_cubes.pth", conf_thresh=0.5):
    device = torch.device("cpu")
    model = FOMOModel(num_classes=4)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # DirectShow for Windows, standard index for Linux/macOS
    import sys
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)

    if not cap.isOpened():
        print(f"Failed to open camera/stream source '{source}'")
        return

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_frame).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).numpy()

        pred_classes = np.argmax(probs, axis=0)
        pred_confs = np.max(probs, axis=0)
        grid_h, grid_w = pred_classes.shape

        for gy in range(grid_h):
            for gx in range(grid_w):
                cls_id = pred_classes[gy, gx]
                conf = pred_confs[gy, gx]

                if cls_id > 0 and conf >= conf_thresh:
                    cx = int((gx + 0.5) * (w / grid_w))
                    cy = int((gy + 0.5) * (h / grid_h))
                    cv2.circle(frame, (cx, cy), 6, CLASS_COLORS.get(cls_id, (255, 255, 255)), -1)
                    cv2.putText(frame, f"{CLASS_NAMES.get(cls_id)} {conf:.2f}",
                                (cx - 25, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        fps = 1.0 / (time.time() - prev_time)
        prev_time = time.time()
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Centroid Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stream(source=0)