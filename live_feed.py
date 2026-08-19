import cv2
import torch
import numpy as np
import time
from train import FOMOModel
from torchvision import transforms

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

    print(f"Opening video source: {source}")
    
    # 1. Open with V4L2 backend
    cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)

    # 2. Force MJPEG compression to prevent select() timeout over USB/IP
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Unable to open camera on source '{source}'.")
        return

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream ended or frame could not be read.")
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

                    color = CLASS_COLORS.get(cls_id, (255, 255, 255))
                    cv2.circle(frame, (cx, cy), 6, color, -1)
                    cv2.putText(
                        frame,
                        f"{CLASS_NAMES.get(cls_id, 'Obj')} {conf:.2f}",
                        (cx - 25, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (255, 255, 255),
                        1
                    )

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Centroid FOMO Stream", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_stream(source=0)