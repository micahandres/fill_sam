from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
from tqdm import tqdm

# bad color red highlight: [236, 154, 124]
# darker deep color red: [225, 115, 75]
# good color green higlight: [162, 214, 206]
# deep color green color: [58, 172, 154]

# Set up SAM model
model_type = "vit_b"
checkpoint_path = "checkpoints/original_checkpoint/sam_vit_b_01ec64 (3).pth"
sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam_model.to("cuda")
predictor = SamPredictor(sam_model)

# Input root directory
root_dir = "data/training_no_hairdryer_shampoo_bottle_wobble_surface*/*"
image_paths = sorted(glob(os.path.join(root_dir, "*.png")))
output_root = "pred_pair_masks"

os.makedirs(output_root, exist_ok=True)

for image_path in tqdm(image_paths, desc="Processing images"):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Skipped invalid image: {image_path}")
        continue

    # Extract sketch and mask
    sketch = image[:, :256, :]
    sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

    # Predict mask
    predictor.set_image(sketch_rgb)
    masks, scores, logits = predictor.predict(
        point_coords=None,
        box=None,
        multimask_output=False
    )
    predicted_mask = masks[0].astype(np.uint8)

    # Create colored overlay
    # color_overlay = sketch_rgb.copy()
    # color_overlay[predicted_mask == 1] = [225, 115, 75] # Darker Red overlay

    # Blend with original
    # alpha = 0.6
    # blended = cv2.addWeighted(sketch_rgb, 1 - alpha, color_overlay, alpha, 0)

    # Save output
    rel_path = os.path.relpath(image_path, "data")
    save_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay for: {rel_path}")

print("٩(ˊᗜˋ*)و All images processed and saved to 'output_predictions_baseline'")
