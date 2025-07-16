# script for combing groundingdino and sam for zero shot segmentaton of sketched objects

# TO DO: 
# (zero shot) groundingdino + sam for ONE image
    # [DONE] have groundingdino create a bounding box around detect object
    # have sam use the bounding box to create a mask around the object
    # display the image with the bounding box and mask

# create a dataset of ONLY the concept sketches -> for zero shot evaluation [can use ggl colab for ease lol]


# == STEP 0: import everything == 
import sys
import os
import torch

"""
# Manually add the GroundingDINO repo to your import path
sys.path.append(os.path.join(os.path.dirname(__file__), "GroundingDINO"))

from groundingdino.util.inference import load_model, load_image, predict, annotate
print("groundingdino modules imported successfully.")
import supervision as sv
import numpy as np
import cv2 
import matplotlib.pyplot as plt
print("OpenCV and Matplotlib imported successfully.")
# device = torch.device('cpu')  # Use CPU for inference

# PART ONE: GroundingDINO
# == step 1: add all paths == 
config_path = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
# checkpoint  is the model weight. we need to download model weight.
checkpoint_path = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
print("Config and checkpoint paths set.")
my_GD_model = load_model(config_path, checkpoint_path) #.to(device)
print("Model loaded successfully.")
IMAGE_PATH = "test_baseline_data/original_concept_all_lines_centered/house_Professional5.png"

# == step 2: add all image and text prompts ==
TEXT_PROMPT = "house"
print("Image path and text prompt set.")
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25
print("Thresholds set.")

# == step 3: load image and predict ==
image_source, myimage = load_image(IMAGE_PATH)
print("Image loaded successfully.")

detected_boxes, accuracy, obj_name = predict(
    model = my_GD_model,
    image = myimage,
    caption = TEXT_PROMPT,
    box_threshold = BOX_THRESHOLD,
    text_threshold = TEXT_THRESHOLD,
    device = "cpu"
)
print(detected_boxes, accuracy, obj_name)

# == step 4: display image with bounding boxes == 
annotated_image = annotate(
    image_source = image_source,
    boxes = detected_boxes,
    logits = accuracy,
    phrases = obj_name
)
print(annotated_image.shape)
sv.plot_image(annotated_image, (10,10))

# == PART 2: SAM ==
# download everything for sam
# import everything for sam
# automatic generate masks based on bounding box input
# use matplotlib to display the image with bounding boxes

# do an evaluation of zero shot sam with produced table of statistics
# == evaluation table of statistics ==
"""