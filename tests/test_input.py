from PIL import Image
import pillow_heif
import os
import pytest
from PIL import Image
import pillow_heif


# Fixture to set up input and output paths
@pytest.fixture
def image_paths():

    # Register HEIF opener
    pillow_heif.register_heif_opener()
    
    input_path = './expense_sorter/input/heic/IMG_1093.HEIC'
    output_path = './expense_sorter/input/png/IMG_1093.png'
    return input_path, output_path

# Test to check if the output file exists
def test_output_file_exists(image_paths):
    """_summary_

    Args:
        image_paths (_type_): _description_
    """
    input_path, output_path = image_paths
    heic_image = Image.open(input_path, formats=["heif"])
    heic_image.save(output_path, "PNG")
    assert os.path.exists(output_path)

    # Remove the output file after testing
    os.remove(output_path)

# Test to check if the conversion is successful
def test_heic_loaded(image_paths):
    """_summary_

    Args:
        image_paths (_type_): _description_
    """
    input_path, output_path = image_paths
    heic_image = Image.open(input_path, formats=["heif"])
    assert heic_image.format == "HEIF"

if __name__ == "__main__":
    pytest.main()
