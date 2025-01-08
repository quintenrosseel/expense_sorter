"""Convert HEIC to PNG, crop image to DETR bounding box,
save cropped image to output directory. """

import os
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PIL import Image
from transformers import pipeline
from transformers.utils import logging
from transformers import AutoImageProcessor, AutoModelForImageSegmentation
from expense_sorter.utils.img import preprocess_heic
from expense_sorter.models.img import (
    get_detr_model,
    detect_bounding_box_detr,
    Models,
    detect_segments,
    detect_bounding_box_opencv,
    crop_image
)

HEIC_DIR = "expense_sorter/input/heic/"
PNG_DIR = "expense_sorter/input/png/" # Intermediate dir for PNG files
OUTPUT_DIR = "expense_sorter/output/" # Final dir for cropped images
MODEL = "bg_removal" # Options: detr_object_detection, opencv, detr_segmentation, bg_removal

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

    for heic_file, input_path_png in processed_files:
        # Load the model
        if MODEL == "detr_object_detection":
            image_processor, model = get_detr_model(
                model_name=Models.DETR_OBJ.value
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
        elif MODEL == "detr_segmentation":
            image_processor, model = get_detr_model(
                model_name=Models.DETR_SEG.value
            )
            logger.info(f"Detecting segments for {input_path_png}")
            # Detect segments using DETR
            panoptic_seg = detect_segments(
                image_path=input_path_png,
                model=model,
                image_processor=image_processor
            )

            plt.imshow(panoptic_seg.astype(np.uint8))
            plt.show()
        elif MODEL == "opencv":
            logger.info(f"Detecting objects for {input_path_png}")

            # Detect objects using OpenCV
            bounding_box = detect_bounding_box_opencv(input_path_png)

            if bounding_box:
                # Unpack the bounding box into the expected format for PIL
                x, y, w, h = bounding_box

                # Load the image using PIL
                input_image = Image.open(input_path_png)

                # Crop the image using the bounding box
                cropped_image = input_image.crop(box=(
                    x,
                    y,
                    x + w,  # right (x + width)
                    y + h   # lower (y + height)
                ))

                file_name = heic_file.split('.')[0].split('/')[-1]
                cropped_output_path = os.path.join(
                    OUTPUT_DIR,
                    f"cropped_{file_name}.png"
                )

                # Save the cropped image
                cropped_image.save(cropped_output_path)
                logger.info(f"Cropped image saved to {cropped_output_path}")
            else:
                raise ValueError(f"No bounding box detected for {input_path_png}")
        elif MODEL == "bg_removal":
            logger.info(f"Detecting objects for {input_path_png}")

            # Check out: https://huggingface.co/briaai/RMBG-1.4
            # Load image
            image = Image.open(input_path_png)
            
            pipe = pipeline(
                 "image-segmentation",
                 model="briaai/RMBG-1.4",
                 trust_remote_code=True
            )
            pillow_mask = pipe(
                 input_path_png,
                 return_mask = True
            ) # outputs a pillow mask
            pillow_image = pipe(
                input_path_png
            ) # applies mask on input and returns a pillow image
            # Save the output image
            file_name = heic_file.split('.')[0].split('/')[-1]
            cropped_output_path = os.path.join(
                OUTPUT_DIR,
                f"cropped_{file_name}.png"
            )

            # Write to file
            pillow_image.save(cropped_output_path)
            logger.info(f"Cropped image saved to {cropped_output_path}")

def convert_png_to_jpg(input_dir: str = PNG_DIR, output_dir: str = OUTPUT_DIR) -> List[Tuple[str, str]]:
    """Convert PNG files to JPG format.
    
    Args:
        input_dir (str): Directory containing PNG files
        output_dir (str): Directory to save JPG files
        
    Returns:
        List[Tuple[str, str]]: List of tuples containing (original_file, output_path)
    """
    processed_files = []
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each PNG file in the input directory
    for file in os.listdir(input_dir):
        if file.lower().endswith('.png'):
            input_path = os.path.join(input_dir, file)
            output_filename = f"{os.path.splitext(file)[0]}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            
            try:
                # Open and convert the image
                with Image.open(input_path) as img:
                    # Create a white background image
                    white_bg = Image.new('RGBA', img.size, 'WHITE')
                    # Paste the image on white background using alpha composite
                    white_bg.paste(img, mask=img)
                    # Convert to RGB
                    rgb_img = white_bg.convert('RGB')
                    # Save as JPG
                    rgb_img.save(output_path, 'JPEG')
                    processed_files.append((file, output_path))
                    logger.info(f"Converted {file} to JPG: {output_path}")
            except Exception as e:
                logger.error(f"Error converting {file}: {str(e)}")
                
    return processed_files

if __name__ == "__main__":
    converted_files = convert_png_to_jpg()
    # main() # Convert from heic to crops

