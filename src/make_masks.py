from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm
import zipfile
import shutil

# === SETUP ===
model_type = "vit_b"
checkpoint_path = "checkpoints/model6_checkpoints/sam_vit_b_01ec64_epoch_10.pth"
sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam_model.to("cuda")
predictor = SamPredictor(sam_model)

# === INPUT & TEMP OUTPUT PATHS ===
input_root = "data/test_only_hairdryer_shampoo_bottle"
temp_output_root = "temp_finetune_masks"
zip_filename = "finetune_masks_compare.zip"

image_paths = sorted(glob(os.path.join(input_root, "*", "*.png")))
os.makedirs(temp_output_root, exist_ok=True)

# === PROCESSING ===
for image_path in tqdm(image_paths, desc="Generating masks"):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipped invalid image: {image_path}")
        continue

    sketch = image[:, :256, :]
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

    predictor.set_image(sketch_rgb)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        box=None,
        multimask_output=True
    )

    if len(masks) == 0:
        print(f"No masks found for: {image_path}")
        continue

    best_mask = masks[np.argmax([np.sum(m) for m in masks])].astype(np.uint8) * 255

    rel_path = os.path.relpath(image_path, input_root)
    rel_path = os.path.splitext(rel_path)[0] + ".png"
    save_path = os.path.join(temp_output_root, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, best_mask)

# === ZIP & CLEANUP ===
with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, _, files in os.walk(temp_output_root):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, temp_output_root)
            zipf.write(file_path, arcname)


print(f"All masks zipped and saved as: {zip_filename}")


