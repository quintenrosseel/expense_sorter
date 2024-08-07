import os
import pillow_heif
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import DetrImageProcessor, DetrForObjectDetection
from typing import List
from transformers.utils import logging
from loguru import logger

def convert_heic_to_png(input_path: str, output_path: str):
    """Convert HEIC file to PNG

    Args:
        input_path (str): Input path to the HEIC file
        output_path (str): Output path to the PNG file
    """
    pillow_heif.register_heif_opener()

    # Do not convert if the output file already exists
    if os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping conversion.")
        return
    else:
        # Open HEIC file and convert to PNG
        heic_image = Image.open(input_path)
        heic_image.save(output_path, "PNG")
        logger.info(f"Converted {input_path} to {output_path}")


def plot_results(pil_img, scores, labels, boxes):
    """_summary_

    Args:
        pil_img (_type_): _description_
        scores (_type_): _description_
        labels (_type_): _description_
        boxes (_type_): _description_
    """
    COLORS = [
        [0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
        [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]
    ]

    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        text = f'{model.config.id2label[label]}: {score:0.2f}'
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))

def detect_bounding_box_detr(image_path, model, image_processor) -> List:
    """_summary_

    Args:
        image_path (_type_): _description_
        model (_type_): _description_
        image_processor (_type_): _description_

    Returns:
        List: _description_
    """
    # Detection for bounding boxes.
    THRESHOLD = 0.0001
    NUM_BOXES = 1

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

def main():
    """
    Convert HEIC files to PNG, 
    crop the image to the bounding box detected by DETR, 
    save the cropped image to the output directory, 
    TODO: apply OCR to the cropped image,
    TODO: save the OCR results to a text file, 
    TOOD: use GPT to get key information out of the OCR results. 
    """
    input_dir_heic = "./expense_sorter/input/heic/"
    input_dir_png = "./expense_sorter/input/png/" # Intermediate dir for PNG files
    output_dir = "expense_sorter/output/" # Final dir for cropped images

    # Load DETR model and image processor (Detection Transformer)
    model_name = "facebook/detr-resnet-50"
    
    # Set Transformers loggger to ERROR (default is INFO)
    logging.set_verbosity_error()
    image_processor = DetrImageProcessor.from_pretrained(
        model_name
    )
    model = DetrForObjectDetection.from_pretrained(
        model_name
    )

    # Get a list of all .heic files in the input directory
    # Defaults to heic files (from iphone)
    files = [
        f for f in os.listdir(input_dir_heic)
        if f.endswith(".HEIC")
    ]

    # Apply crop_pdf to each file and save it in the output directory with the same name
    for heic_file in files:
        input_path_heic = os.path.join(input_dir_heic, heic_file)
        input_path_png = os.path.join(input_dir_png, heic_file.split('.')[0] + ".png")

        # Ensure the PNG file exists
        convert_heic_to_png(
            input_path_heic,
            input_path_png
        )

        input_path_png = os.path.join(
            input_dir_png, 
            f"{heic_file.split('.')[0]}.png"
        )

        # # Detect bounding box using DETR
        bboxes = detect_bounding_box_detr(
            image_path=input_path_png,
            model=model,
            image_processor=image_processor
        )

        if len(bboxes) > 0:
            for i, bbox in enumerate(bboxes):
                # output_path = os.path.join(output_dir, f"{i}_{file}")
                x_min, y_min, x_max, y_max = bbox.int().tolist()
                logger.info(
                    f"Bounding box for {input_path_png}: x_min={x_min}, y_min={y_min}, "
                    f"x_max={x_max}, y_max={y_max}"
                )

                # Crop the image to the bounding box
                image = Image.open(input_path_png)
                cropped_image = image.crop((x_min, y_min, x_max, y_max))
                cropped_output_path = os.path.join(
                    output_dir, 
                    f"cropped_{i}_{heic_file.split('.')[0]}.png"
                )
                cropped_image.save(cropped_output_path)
                
                logger.info(f"Cropped image saved to {cropped_output_path}")
        else:
            logger.debug(f"No bounding box found for {input_path_png}")
            image = Image.open(input_path_png)
            not_cropped_output_path = os.path.join(
                output_dir, 
                f"original_{heic_file.split('.')[0]}.png"
            )
            image.save(not_cropped_output_path)

if __name__ == "__main__":
    main()
