import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
import intel_extension_for_pytorch as ipex

# 1. Model Definition (Centroid Grid Detector / FOMO)
class FOMOModel(nn.Module):
    def __init__(self, num_classes=3): # e.g., 3 cube colors
        super().__init__()
        # MobileNetV2 truncated to stride 8
        features = mobilenet_v2(weights="DEFAULT").features
        self.backbone = nn.Sequential(*features[:7])
        
        # 1x1 Conv Head -> outputs (C + 1) channels (0 = background)
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes + 1, kernel_size=1)
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

# 2. Centroid Target Converter
def create_centroid_mask(boxes, labels, grid_size=(12, 12), scale_factor=8):
    mask = torch.zeros(grid_size, dtype=torch.long)
    for box, label in zip(boxes, labels):
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        gx = int(cx // scale_factor)
        gy = int(cy // scale_factor)
        if 0 <= gx < grid_size[1] and 0 <= gy < grid_size[0]:
            mask[gy, gx] = label
    return mask

# 3. Training Loop Setup
def train():
    device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
    model = FOMOModel(num_classes=3).to(device)
    
    # Background class (0) is dominant; weight positive classes higher
    class_weights = torch.tensor([0.05, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Optimize model & optimizer for Intel hardware
    model, optimizer = ipex.optimize(model, optimizer=optimizer)

    print(f"Training initialized on device: {device}")
    # Add DataLoader loop and forward/backward passes here

if __name__ == "__main__":
    train()