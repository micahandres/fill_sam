# model 7 finetune.py
# bce and dice loss, with a smaller learning rate and more epochs

import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from segment_anything import SamPredictor, sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
from torch.utils.data import Dataset
from statistics import mean
from glob import glob
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# To view training logs, run:
# tensorboard --logdir=runs/model7_runs/model7_vit_b_512_logs

# Settings for SAM model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_type = "vit_b"
checkpoint_path = os.path.join(BASE_DIR, "checkpoints", "model7_checkpoints","sam_vit_b_01ec64.pth")
desired_size = (256, 256)
train_root = os.path.join(BASE_DIR, "data", "training_no_hairdryer_shampoo_bottle_wobble_surface")
val_root = os.path.join(BASE_DIR, "data", "validation_only_wobble_surface")
batch_size = 8
num_epochs = 20
lr = 1e-5
wd = 1e-4
save_path = os.path.join(BASE_DIR, "checkpoints", "model7_checkpoints", "sam_vit_b_01ec64.pth")
device = torch.device("cuda")

log_dir = os.path.join(BASE_DIR, "runs", "model7_runs", "model7_vit_b_512_logs")
writer = SummaryWriter(log_dir=log_dir)

class SketchMaskDataset(Dataset):
    def __init__(self, image_paths, sam_model, desired_size=desired_size, device=device):
        self.image_paths = image_paths
        self.desired_size = desired_size
        self.device = device
        self.preprocess_transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        self.sam_model = sam_model

    def __len__(self):
        return len(self.image_paths)

    def load_sketch_and_mask(self, path):
        img = cv2.imread(path)
        sketch = img[:, :256, :]
        mask = img[:, 256:, :]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask = (mask > 0).astype(np.uint8)
        return sketch, mask

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        sketch, mask = self.load_sketch_and_mask(path)
        sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)
        input_image = self.preprocess_transform.apply_image(sketch_rgb)
        input_image_torch = torch.as_tensor(input_image, device=self.device).permute(2, 0, 1).contiguous()
        input_image_processed = self.sam_model.preprocess(input_image_torch.unsqueeze(0)).squeeze(0)
        mask_tensor = torch.from_numpy(mask).float().unsqueeze(0).to(self.device)
        return {
            'image': input_image_processed,
            'mask': mask_tensor,
            'original_image_size': torch.tensor(sketch_rgb.shape[:2]),
            'input_size': torch.tensor(input_image_processed.shape[-2:]),
            'path': path
        }

def threshold(preds, thresh=0.5):
    return (preds > thresh).float()

def calculate_accuracy(preds, targets):
    return (threshold(torch.sigmoid(preds)) == targets).float().mean().item()

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam_model.to(device)
sam_model.train()

train_paths = sorted(glob(os.path.join(train_root, "**/*.png"), recursive=True))
val_paths = sorted(glob(os.path.join(val_root, "**/*.png"), recursive=True))
print(f"Train images: {len(train_paths)} | Val images: {len(val_paths)}")

train_dataset = SketchMaskDataset(train_paths, sam_model, desired_size=desired_size, device=device)
val_dataset = SketchMaskDataset(val_paths, sam_model, desired_size=desired_size, device=device)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
val_loader = DataLoader(val_dataset, batch_size=1, pin_memory=False)

# Custom Loss 
class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(1)
        dice = (2. * intersection + self.eps) / (inputs.sum(1) + targets.sum(1) + self.eps)
        return 1 - dice.mean()

class BCEDiceLoss(torch.nn.Module):
    def __init__(self, alpha=0.5):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.alpha = alpha

    def forward(self, inputs, targets):
        return self.alpha * self.bce(inputs, targets) + (1 - self.alpha) * self.dice(inputs, targets)

optimizer = torch.optim.Adam(sam_model.mask_decoder.parameters(), lr=lr, weight_decay=wd)
loss_fn = BCEDiceLoss(alpha=0.5)
global_step = 0
best_val_loss = float('inf')

for epoch in range(num_epochs):
    sam_model.train()
    train_losses, train_accuracies = [], []

    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Training]"):
        input_image = batch['image'].to(device)
        gt_mask = batch['mask'].to(device)
        input_size = batch['input_size']
        original_size = batch['original_image_size']

        with torch.no_grad():
            image_embedding = sam_model.image_encoder(input_image)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)

        low_res_masks, _ = sam_model.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=sam_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        upscaled_masks = []
        for i in range(input_image.shape[0]):
            input_sz = tuple(map(int, input_size[i]))
            orig_size = tuple(map(int, original_size[i]))
            mask = sam_model.postprocess_masks(low_res_masks[i:i+1], input_sz, orig_size).to(device)
            upscaled_masks.append(mask)
        upscaled_masks = torch.cat(upscaled_masks, dim=0)

        loss = loss_fn(upscaled_masks, gt_mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc = calculate_accuracy(upscaled_masks, gt_mask)
        train_losses.append(loss.item())
        train_accuracies.append(acc)

        writer.add_scalar("Train/Loss", loss.item(), global_step)
        writer.add_scalar("Train/Accuracy", acc, global_step)
        global_step += 1

    print(f"[Epoch {epoch}] Train Loss: {mean(train_losses):.4f} | Accuracy: {mean(train_accuracies):.4f}")

    sam_model.eval()
    val_loss, val_accuracy, count = 0.0, 0.0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Validation]"):
            input_image = batch['image'].to(device)
            gt_mask = batch['mask'].to(device)
            input_size = batch['input_size']
            original_size = batch['original_image_size']

            image_embedding = sam_model.image_encoder(input_image)
            sparse_embeddings, dense_embeddings = sam_model.prompt_encoder(points=None, boxes=None, masks=None)

            low_res_masks, _ = sam_model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            input_sz = tuple(input_size[0].tolist())
            orig_size = tuple(original_size[0].tolist())

            upscaled_mask = sam_model.postprocess_masks(low_res_masks, input_sz, orig_size).to(device)

            loss = loss_fn(upscaled_mask, gt_mask)
            acc = calculate_accuracy(upscaled_mask, gt_mask)

            val_loss += loss.item()
            val_accuracy += acc
            count += 1

    mean_val_loss = val_loss / count
    mean_val_accuracy = val_accuracy / count

    print(f"[Epoch {epoch}] Val Loss: {mean_val_loss:.4f} | Val Accuracy: {mean_val_accuracy:.4f}")

    writer.add_scalar("Val/Loss", mean_val_loss, epoch)
    writer.add_scalar("Val/Accuracy", mean_val_accuracy, epoch)

    if mean_val_loss < best_val_loss:
        best_val_loss = mean_val_loss
        save_checkpoint(sam_model, save_path)
        print(f"[Epoch ૮˶ᵔᵕᵔ˶ა {epoch}] Best model updated and saved to {save_path}")

    epoch_save_path = save_path.replace(".pth", f"_epoch_{epoch+1}.pth")
    save_checkpoint(sam_model, epoch_save_path)
    print(f"[Epoch (˶°ㅁ°)!! {epoch}] Saved model to {epoch_save_path}")

    torch.cuda.empty_cache()

writer.close()