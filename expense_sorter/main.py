"""Convert HEIC to PNG, crop image to DETR bounding box,
save cropped image to output directory. """

import os
from typing import Dict, List

import numpy as np
from loguru import logger
from PIL import Image
from transformers.utils import logging
from expense_sorter.utils.img import preprocess_heic
from expense_sorter.models.detr import (
    get_detr_object_detection_model,
    detect_bounding_box_detr,
    Models
)

HEIC_DIR = "expense_sorter/input/heic/"
PNG_DIR = "expense_sorter/input/png/" # Intermediate dir for PNG files
OUTPUT_DIR = "expense_sorter/output/" # Final dir for cropped images
MODEL = "detr_object_detection" # Or detr_object_detection

def main():
    """
    Convert HEIC files to PNG, 
    crop the image to the bounding box detected by DETR, 
    save the cropped image to the output directory, 
    TODO: apply OCR to the cropped image,
    TODO: save the OCR results to a text file, 
    TODO: use GPT to get key information out of the OCR results. 
    """

    # Set Transformers loggger to ERROR (default is INFO)
    logging.set_verbosity_error()

    processed_files = preprocess_heic(
        input_dir=HEIC_DIR,
        processed_dir=PNG_DIR
    )

    # Load the model
    if MODEL == "detr_object_detection":
        image_processor, model = get_detr_object_detection_model(
            model_name=Models.DETR_OBJ.value
        )
    elif MODEL == "detr_segmentation":
        image_processor, model = get_detr_object_detection_model(
            model_name=Models.DETR_SEG.value
        )

    for heic_file, input_path_png in processed_files:
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
                    OUTPUT_DIR,
                    f"cropped_{i}_{heic_file.split('.')[0]}.png"
                )
                cropped_image.save(cropped_output_path)
                logger.info(f"Cropped image saved to {cropped_output_path}")
        else:
            # Open original image
            image = Image.open(
                os.path.join(
                    os.getcwd(),
                    input_path_png
                )
            )

            file_name: str = f"{heic_file.split('.')[0].split('/')[-1]}"
            # Save original image with prefix
            not_cropped_output_path = os.path.join(
                OUTPUT_DIR,
                f"original_{file_name}.png"
            )
            logger.debug(
                "No bounding box found for {}, saving to {}",
                input_path_png,
                not_cropped_output_path
            )
            image.save(not_cropped_output_path)

if __name__ == "__main__":
    main()
