from PIL import Image
import os

def get_tiff_image_size(directory):
    # Get a list of TIFF files in the directory
    tiff_files = [filename for filename in os.listdir(directory) if filename.lower().endswith('.tiff')]

    if not tiff_files:
        return None  # No TIFF files found in the directory

    # Open the first TIFF file
    first_tiff_file_path = os.path.join(directory, tiff_files[0])
    with Image.open(first_tiff_file_path) as img:
        # Get the size of the first image in the stack
        width, height = img.size
        
        # Get the number of frames in the TIFF stack
        stack_size = img.n_frames

    return height, width, stack_size
