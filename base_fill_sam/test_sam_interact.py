from segment_anything import sam_model_registry, SamPredictor
import torch
import cv2
import matplotlib.pyplot as plt

model_type = "vit_b"
checkpoint_path = "checkpoints/model13_checkpoints/sam_vit_b_01ec64_epoch_24.pth"  # saved fine-tuned model

sam_model = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam_model.to("cuda")
predictor = SamPredictor(sam_model)

image_path = "data2/test_only_student9_Professional4/original_concept_all_lines_centered/shampoo_bottle_Professional4.png"
image = cv2.imread(image_path)
sketch = image[:, :256, :]
mask = image[:, 256:, :]
sketch_rgb = cv2.cvtColor(sketch, cv2.COLOR_BGR2RGB)

predictor.set_image(sketch_rgb)
masks, scores, logits = predictor.predict(
    point_coords=None,
    box=None,
    multimask_output=False
)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Show original sketch
axes[0].imshow(image)
axes[0].set_title("Sketch")
axes[0].axis("off")

# Show predicted mask
axes[1].imshow(masks[0], cmap="gray")
axes[1].set_title("Predicted Mask")
axes[1].axis("off")

plt.tight_layout()
plt.tight_layout()
plt.savefig("prediction_result.png")
print("٩(ˊᗜˋ*)و Saved to prediction_result.png")

