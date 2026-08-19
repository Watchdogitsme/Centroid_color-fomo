import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2
from torchvision import transforms

# 1. Custom Dataset for Centroid Grid Detection
class ConveyorCentroidDataset(Dataset):
    def __init__(self, csv_file, img_size=(96, 96), scale_factor=8):
        self.df = pd.read_csv(csv_file)
        self.img_paths = self.df["image_path"].unique()
        self.img_size = img_size
        self.scale_factor = scale_factor
        self.grid_size = (img_size[0] // scale_factor, img_size[1] // scale_factor)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = img.shape[:2]

        boxes = self.df[self.df["image_path"] == img_path][["x_min", "y_min", "x_max", "y_max", "label_id"]].values

        # Build ground-truth grid mask (0 = background)
        mask = np.zeros(self.grid_size, dtype=np.int64)
        for b in boxes:
            x_min, y_min, x_max, y_max, label_id = b
            # Scale centroids to model input resolution
            cx = ((x_min + x_max) / 2.0) * (self.img_size[1] / orig_w)
            cy = ((y_min + y_max) / 2.0) * (self.img_size[0] / orig_h)
            
            gx = int(cx // self.scale_factor)
            gy = int(cy // self.scale_factor)
            
            if 0 <= gx < self.grid_size[1] and 0 <= gy < self.grid_size[0]:
                mask[gy, gx] = int(label_id)

        tensor_img = self.transform(img)
        tensor_mask = torch.tensor(mask, dtype=torch.long)
        return tensor_img, tensor_mask

# 2. FOMO Architecture
class FOMOModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        features = mobilenet_v2(weights="DEFAULT").features
        self.backbone = nn.Sequential(*features[:7])  # Cutoff at 1/8 stride
        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, num_classes + 1, kernel_size=1)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

# 3. Training Loop
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ConveyorCentroidDataset("data/annotations.csv", img_size=(96, 96))
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = FOMOModel(num_classes=4).to(device)
    
    # Weight positive classes to handle background imbalance
    weights = torch.tensor([0.05, 1.0, 1.0, 1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 25
    print(f"Starting training for {epochs} epochs on {device}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), "fomo_cubes.pth")
    print("Model weights saved to fomo_cubes.pth")

if __name__ == "__main__":
    main()