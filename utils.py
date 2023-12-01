import os
import tifffile


def get_tiff_image_size(directory):
    # Get a list of TIFF files in the directory
    tiff_files = [filename for filename in os.listdir(directory) if filename.endswith('.TIFF')]

    if not tiff_files:
        return None  # No TIFF files found in the directory

    # Read the first TIFF file
    first_tiff_file = tifffile.imread(os.path.join(directory, tiff_files[0]))

    # Extract the shape of the first image in the stack
    depth, height, width = first_tiff_file.shape

    return height, width, depth