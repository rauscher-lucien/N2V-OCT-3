import os
import glob
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
from utils import get_tiff_image_size
import matplotlib.pyplot as plt
import logging

class Dataset3D(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        logging.info(self.data_dir)
        self.transform = transform
        self.file_list = sorted(glob.glob(os.path.join(data_dir, '*.tiff')))

        self.height, self.width, self.depth = get_tiff_image_size(self.data_dir)

    def __getitem__(self, index):
        file_index = index // self.depth
        stack_index = index % self.depth

        full_path = self.file_list[file_index]
        filename = os.path.basename(full_path)

        with Image.open(full_path) as img:
            img.seek(stack_index)
            data = np.array(img, dtype=np.float32) / 65535.0  # Normalize to [0, 1]
            data = transforms.ToTensor()(data)

        data = data.squeeze(0)   

        if self.transform:
            data = self.transform(data)

        return data


    def __len__(self):
        return len(self.file_list * self.depth)

    