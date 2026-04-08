import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2 
import numpy as np

# ==========================================
# 0. CONFIGURATION & PATHS (Clear & Centralized)
# ==========================================
CONFIG = {
    "RAW_FRAMES_FOLDER": r"D:\Image_Segmentation_task\Cat_frames",
    "OUTPUT_DIR": r"D:\Image_Segmentation_task\Cat_segmentation_results",            
    "NUM_SEGMENTS": 2,
    "EPOCHS": 200,
    "PATIENCE": 15,                                                # For early stopping
    "IMG_SIZE": (128, 128)
}

# Auto-create necessary output folders
os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
CHECKPOINT_PATH = os.path.join(CONFIG["OUTPUT_DIR"], "best_model.pth")
MASKS_DIR = os.path.join(CONFIG["OUTPUT_DIR"], "segmented_masks")
os.makedirs(MASKS_DIR, exist_ok=True)

# ==========================================
# 1. DATASET SETUP
# ==========================================
class SequenceDataset(Dataset):
    def __init__(self, folder_path, img_size=CONFIG["IMG_SIZE"]):
        self.folder_path = folder_path
        self.image_files = sorted([f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), img_path # Returning path for side-by-side video later

# ==========================================
# 2. THE N-CUT LOSS
# ==========================================
class SoftNCutLoss(nn.Module):
    def __init__(self):
        super(SoftNCutLoss, self).__init__()

    def forward(self, masks, images):
        b, k, h, w = masks.shape
        masks = masks.view(b, k, -1)
        
        intersection = torch.sum(masks * masks, dim=-1)
        total = torch.sum(masks, dim=-1)
        loss = 1 - torch.mean(intersection / (total + 1e-6))
        return loss

# ==========================================
# 3. W-NET ARCHITECTURE (Increased Complexity)
# ==========================================
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class WNet(nn.Module):
    def __init__(self, num_segments=2): 
        super().__init__()
        # Added a 128-channel layer for deeper feature extraction
        self.enc1 = UNetBlock(3, 64)
        self.enc2 = UNetBlock(64, 128) 
        self.enc3 = nn.Conv2d(128, num_segments, 1)
        
        self.dec1 = UNetBlock(num_segments, 128)
        self.dec2 = UNetBlock(128, 64)
        self.dec3 = nn.Conv2d(64, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        segmentation = F.softmax(self.enc3(e2), dim=1)
        
        d1 = self.dec1(segmentation)
        d2 = self.dec2(d1)
        reconstruction = self.dec3(d2)
        return segmentation, reconstruction

# ==========================================
# GRADIENT CHECKER (Industry Standard)
# ==========================================
def check_gradient_health(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

# ==========================================
# 4. TRAINING LOOP (Early Stopping & Clear Logs)
# ==========================================
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INIT] Training on {device} | Output to {CONFIG['OUTPUT_DIR']}")

    dataset = SequenceDataset(CONFIG["RAW_FRAMES_FOLDER"])
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = WNet(num_segments=CONFIG["NUM_SEGMENTS"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    ncut_loss_func = SoftNCutLoss()
    reconstruction_loss_func = nn.MSELoss()

    # Tracking for plots
    history = {"Total": [], "Shape_Loss": [], "Recon_Loss": [], "Grad_Norm": []}
    
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(CONFIG["EPOCHS"]):
        ep_total, ep_ncut, ep_rec, ep_grad = 0, 0, 0, 0
        
        for images, _ in dataloader:
            images = images.to(device)
            masks, recons = model(images)
            
            loss_ncut = ncut_loss_func(masks, images)
            loss_rec = reconstruction_loss_func(recons, images)
            
            class_usage = torch.mean(masks, dim=(0, 2, 3)) 
            loss_usage = torch.mean(torch.relu(0.02 - class_usage))
            
            total_loss = loss_ncut + (loss_rec * 1000) + (loss_usage * 500)
            
            optimizer.zero_grad()
            total_loss.backward()
            
            # Check gradients before stepping
            grad_norm = check_gradient_health(model)
            if grad_norm < 1e-4:
                print("  [WARNING] Vanishing Gradient Detected! Model may stop learning.")
                
            optimizer.step()
            
            ep_total += total_loss.item()
            ep_ncut += loss_ncut.item()
            ep_rec += loss_rec.item()
            ep_grad += grad_norm

        # Calculate averages for logging
        avg_total = ep_total / len(dataloader)
        avg_ncut = ep_ncut / len(dataloader)
        avg_rec = ep_rec / len(dataloader)
        
        history["Total"].append(avg_total)
        history["Shape_Loss"].append(avg_ncut)
        history["Recon_Loss"].append(avg_rec)
        history["Grad_Norm"].append(ep_grad / len(dataloader))

        # Clear Terminal Output
        print(f"Epoch [{epoch+1:02d}/{CONFIG['EPOCHS']}] | Total Loss: {avg_total:.4f} | Shape Loss: {avg_ncut:.4f} | Recon Loss: {avg_rec:.4f} | Grad Norm: {history['Grad_Norm'][-1]:.4f}")

        # Checkpointing & Early Stopping
        if avg_total < best_loss:
            best_loss = avg_total
            epochs_no_improve = 0
            torch.save(model.state_dict(), CHECKPOINT_PATH)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= CONFIG["PATIENCE"]:
                print(f"\n[INFO] Early stopping triggered. No improvement for {CONFIG['PATIENCE']} epochs.")
                break
                
    # Save the Loss Curve Plot
    save_training_curves(history)
    print(f"\n[DONE] Training Complete. Best model saved to: {CHECKPOINT_PATH}")

def save_training_curves(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history["Shape_Loss"], label="Shape Loss (N-Cut)")
    plt.plot(history["Recon_Loss"], label="Reconstruction Loss (MSE)")
    plt.title("Unsupervised Training Metrics (Lower is Better)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(CONFIG["OUTPUT_DIR"], "training_curve.png")
    plt.savefig(plot_path)
    plt.close()

# ==========================================
# 5. INFERENCE & VISUALIZATION
# ==========================================
def save_segmented_images():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[INFO] Generating Masks from Best Checkpoint...")
    
    model = WNet(num_segments=CONFIG["NUM_SEGMENTS"]).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()

    dataset = SequenceDataset(CONFIG["RAW_FRAMES_FOLDER"])

    with torch.no_grad():
        for i, (img_tensor, _) in enumerate(dataset):
            input_batch = img_tensor.unsqueeze(0).to(device)
            masks, _ = model(input_batch)
            
            seg_map = torch.argmax(masks, dim=1).squeeze().cpu().numpy()
            
            plt.figure(figsize=(5, 5))
            plt.imshow(seg_map, cmap='magma', vmin=0, vmax=CONFIG["NUM_SEGMENTS"]-1) 
            plt.axis('off')
            plt.savefig(os.path.join(MASKS_DIR, f"frame_{i:04d}.png"), bbox_inches='tight', pad_inches=0)
            plt.close()
            
    print(f"[DONE] Saved frames to '{MASKS_DIR}'")

# ==========================================
# 6. COMPILE SIDE-BY-SIDE VIDEO
# ==========================================
def create_video(fps=15):
    print("\n[INFO] Compiling Side-by-Side Video...")
    video_name = os.path.join(CONFIG["OUTPUT_DIR"], "SideBySide_Result.mp4")
    
    dataset = SequenceDataset(CONFIG["RAW_FRAMES_FOLDER"])
    mask_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith(".png")])
    
    if not mask_files: 
        print("No masks found to compile.")
        return

    # Read the first pair to get dimensions
    sample_mask = cv2.imread(os.path.join(MASKS_DIR, mask_files[0]))
    h, w, _ = sample_mask.shape
    
    # Video writer (width is doubled because of side-by-side)
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))

    for i in range(len(mask_files)):
        # Load generated mask
        mask_img = cv2.imread(os.path.join(MASKS_DIR, mask_files[i]))
        
        # Load and resize original frame to match the mask size
        _, orig_path = dataset[i]
        orig_img = cv2.imread(orig_path)
        orig_img = cv2.resize(orig_img, (w, h))
        
        # Concatenate horizontally (Original on Left, Mask on Right)
        combined_frame = cv2.hconcat([orig_img, mask_img])
        video.write(combined_frame)

    video.release()
    print(f"[DONE] Professional Video saved as '{video_name}'")

# ==========================================
# RUN SCRIPT
# ==========================================
if __name__ == "__main__":
    train_model()
    save_segmented_images()
    create_video(fps=15)
