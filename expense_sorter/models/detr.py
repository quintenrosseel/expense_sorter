"""Model for object detection & segmentation using DETR."""

from typing import Tuple, Dict, List
from enum import Enum
from PIL import Image
from loguru import logger
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    PreTrainedModel
)

import torch

class Models(Enum):
    """Available models for image analysis"""
    # DETR model and image processor (Detection Transformer)
    DETR_OBJ = "facebook/detr-resnet-50"
    DETR_SEG = "facebook/detr-resnet-50-panoptic"

def get_detr_object_detection_model(
        model_name: str) -> Tuple[object, PreTrainedModel]:
    """_summary_

    Args:
        model_name (_type_): _description_

    Returns:
        PreTrainedModel: _description_
    """
    image_processor = DetrImageProcessor.from_pretrained(
        model_name
    )
    model = DetrForObjectDetection.from_pretrained(
        model_name
    )
    return image_processor, model


def detect_bounding_box_detr(
        image_path: str,
        model: PreTrainedModel,
        image_processor: DetrImageProcessor) -> List[Dict]:
    """_summary_

    Args:
        image_path (_type_): Path of the image to process
        model (_type_): Model to use for inference
        image_processor (_type_): Image processor to preprocess the image

    Returns:
        List: List of bounding boxes
    """
    # Detection for bounding boxes.
    THRESHOLD = 0.0001
    NUM_BOXES = 1

    logger.info(f"Loading image from {image_path}")

    # Open the image with PIL
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(
        images=image,
        return_tensors="pt"
    )
    logger.info(f"Processing {image_path} at threshold {THRESHOLD}")
    logger.debug(f"Encoder Keys: {inputs.keys()}")
    logger.debug(f"Encoder Shape: {inputs['pixel_values'].shape}")

    # Forward pass through the model
    # using torch.no_grad() will save us memory: no gradients need at inference time
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert outputs (logits) to bounding boxes
    # The target sizes should be in the format (height, width)
    width, height = image.size
    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=[(height, width)],
        threshold=THRESHOLD
    )

    if not results:
        return []  # No detections found

    # Assume the largest detected box is the expense ticket
    boxes = results[0]['boxes']
    scores = results[0]['scores']

    # Filter out boxes with low confidence scores
    high_conf_boxes = boxes[scores > THRESHOLD]

    logger.info(f"Found {len(high_conf_boxes)} box(es) with confidence > {THRESHOLD}")

    if len(high_conf_boxes) == 0:
        return []

    # Get the largest box
    # largest_box = high_conf_boxes[0]
    return high_conf_boxes[0:NUM_BOXES]
