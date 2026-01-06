# Expense Sorter

## Overview

Expense Sorter is a Python-based tool designed to help you organize raw expenses (as iphone images files) efficiently. It processes input files, applies necessary transformations, and saves the output in a specified directory.

## Features

- **File Processing**: Automatically processes `.heic` files from the input directory.
- **Output Management**: Saves processed files to a specified output directory.
- **Customizable**: Easily modify the script to fit your specific needs.
- **Custom Deep Learning Model**: Uses a DETR model for object detection, as demoed by Niels Rogge [over here](https://colab.research.google.com/github/NielsRogge/Transformers-Tutorials/blob/master/DETR/DETR_minimal_example.ipynb#scrollTo=-Wc92cWK-Aas)

## Requirements

- Torch 2.0.0
- Pillow for image processing. 
- Python 3.10
- PyMuPDF (for PDF processing)
- Hugging Face transformers for loading 
- Poetry (for dependency management)

# Running Tests 
```zsh 
    pytest . 
```

# Running
- Ensure that the `input/heic` folder contains your image files you want to process. 
- Next, in the root level directory (containing your environment), run: 
```python
python3 ./expense_sorter/main.py
```
    - The script will take a while to start up, and log. 
    - The `main()` script transforms heic files to cropped versions of those images.
        - First transforms to PNG files
        - Next, transforms to cropped versions using the BG removal model. 
- Typically, the output needs to be JPEG as PNG isn't supported. 
    - copy the images to input/png
    - Uncomment the `main` script and enable the `convert_png_to_jpg` script in the `main.py` file. 
    - Run again. 