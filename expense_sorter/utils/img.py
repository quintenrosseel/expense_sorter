"""Utility classes for dealing with images"""
import os

from typing import List, Tuple

from loguru import logger
from PIL import Image

import pillow_heif
import matplotlib.pyplot as plt

def convert_heic_to_png(input_path: str, output_path: str) -> None:
    """Convert HEIC file to PNG

    Args:
        input_path (str): Input path to the HEIC file
        output_path (str): Output path to the PNG file
    """
    pillow_heif.register_heif_opener()

    # Do not convert if the output file already exists
    if os.path.exists(output_path):
        logger.info(f"Output file {output_path} already exists. Skipping conversion.")
    else:
        # Open HEIC file and convert to PNG
        heic_image = Image.open(input_path)
        heic_image.save(output_path, "PNG")
        logger.info(f"Converted {input_path} to {output_path}")


def preprocess_heic(
        input_dir: str,
        processed_dir: str) -> List[Tuple[str, str]]:
    """Convert HEIC files to PNG, 
    crop the image to the bounding box detected by DETR, 
    save the cropped image to the output directory,"""

    # Get a list of all .heic files in the input directory
    # Defaults to heic files (from iphone)
    files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".HEIC")
    ]

    processed_paths: List[Tuple[str, str]] = []

    # Apply crop_pdf to each file and save it in the output directory with the same name
    for heic_file in files:
        input_path_heic = os.path.join(os.
            getcwd(),
            input_dir,
            heic_file
        )
        input_path_png = os.path.join(
            os.getcwd(),
            processed_dir,
            heic_file.split('.')[0] + ".png"
        )

        # Ensure the PNG file exists
        convert_heic_to_png(
            input_path_heic,
            input_path_png
        )
        processed_paths.append((
            os.path.join( # png_file path
                input_dir,
                heic_file
            ),
            os.path.join( # png_file path
                processed_dir,
                f"{heic_file.split('.')[0]}.png"
            )
        ))
    return processed_paths

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
