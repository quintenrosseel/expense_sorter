"""Model for object detection & segmentation using DETR."""

import io
import itertools
from enum import Enum
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy
import numpy as np
import seaborn as sns
import torch
from loguru import logger
from PIL import Image
from transformers import (DetrForObjectDetection, DetrForSegmentation,
                          DetrImageProcessor, PreTrainedModel)
from transformers.image_transforms import rgb_to_id


class Models(Enum):
    """Available models for image analysis"""
    # DETR model and image processor (Detection Transformer)
    DETR_OBJ = "facebook/detr-resnet-50"
    DETR_SEG = "facebook/detr-resnet-50-panoptic"
    CV2_DET = "opencv"

def get_detr_model(
        model_name: str) -> Tuple[object, PreTrainedModel]:
    """_summary_

    Args:
        model_name (str): Hugging Face model name

    Returns:
        PreTrainedModel: Wrapped DETR model
    """
    image_processor = DetrImageProcessor.from_pretrained(
        model_name
    )
    if model_name == Models.DETR_SEG.value:
        model = DetrForSegmentation.from_pretrained(
            model_name
        )
        return image_processor, model
    elif model_name == Models.DETR_OBJ.value:
        model = DetrForObjectDetection.from_pretrained(
            model_name
        )
        return image_processor, model
    else:
        raise ValueError(f"Model {model_name} not supported")

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

def detect_segments(
    image_path: str,
    model: PreTrainedModel,
    image_processor: DetrImageProcessor) -> List[np.ndarray]:
    """Note: this currently just returns the panoptic segmentation images. 
    The model should be fine tuned https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DETR/Fine_tuning_DetrForSegmentation_on_custom_dataset_end_to_end_approach.ipynb"""
            
    image = Image.open(image_path).convert("RGB")
    encoding = image_processor(
        image,
        return_tensors="pt"
    ) # type: ignore
    
    outputs = model(**encoding)
    processed_sizes = torch.as_tensor(
        encoding['pixel_values'].shape[-2:]
    ).unsqueeze(0)
    result = image_processor.post_process_panoptic(
        outputs,
        processed_sizes
    )[0]

    palette = itertools.cycle(
        sns.color_palette()
    )

    # The segmentation is stored in a special-format png
    panoptic_seg = Image.open(
        io.BytesIO(result['png_string'])
    )
    panoptic_seg = numpy.array(
        panoptic_seg,
        dtype=numpy.uint8).copy()
    
    # We retrieve the ids corresponding to each mask
    panoptic_seg_id = rgb_to_id(
        panoptic_seg
    )
    
    for id in numpy.unique(panoptic_seg_id):
        mask = panoptic_seg_id == id
        color: Tuple[float, float, float] = next(palette)
        np_color = np.asarray(color) * 255
        panoptic_seg[mask] = np_color
        logger.info(f"Coloring mask {id} with color {color}")

    return panoptic_seg.astype(np.uint8)

def detect_bounding_box_opencv(image_path: str) -> List[int]:
    """Detects the bounding box of the largest contour using OpenCV.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        List[int]: Bounding box coordinates [x, y, w, h].
    """
    logger.info(f"Loading image from {image_path}")

    # Load image with alpha channel if present
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        logger.error(f"Failed to load image from {image_path}.")
        return []

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (50, 50), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, threshold1=100, threshold2=150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Debug: Draw all contours
    debug_image = image.copy()
    cv2.drawContours(debug_image, contours, -1, (0, 255, 0), 3)
    debug_path = image_path.replace('.png', '_contours_debug.png')
    cv2.imwrite(debug_path, debug_image)
    logger.info(f"Contour debug image saved to {debug_path}")

    # Sort contours by area and grab the largest one
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    if not contours:
        logger.error("No contours found.")
        return []

    largest_contour = contours[0]

    # Get the bounding box around the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    logger.info(f"Bounding box found: x={x}, y={y}, w={w}, h={h}")

    return [x, y, w, h]
def crop_image(image_path: str, bounding_box: List[int]) -> Image:
    """Crops the image using the provided bounding box coordinates.

    Args:
        image_path (str): Path to the image file.
        bounding_box (List[int]): List of coordinates [x, y, w, h].

    Returns:
        Image: Cropped image as a PIL Image.
    """
    if not bounding_box:
        logger.error("Empty bounding box provided. Returning the original image.")
        return Image.open(image_path)

    image = Image.open(image_path)
    x, y, w, h = bounding_box
    cropped_image = image.crop((x, y, x + w, y + h))

    logger.info("Image cropped successfully.")
    return cropped_image