import glob
import torch

from utils import *


class Dataset3D(torch.utils.data.Dataset):

    def __init__(self, data_dir, transform=None):

        self.data_dir = data_dir
        self.transform = transform
        self.file_list = sorted(glob.glob(os.path.join(data_dir, '*.tiff')))

        self.height, self.width, self.depth = get_tiff_image_size(self.data_dir)

    def __getitem__(self, index):

        file_index = index//self.depth
        stack_index = index % self.depth

        filename = self.file_list[file_index]

        data = tifffile.imread(os.path.join(self.data_dir, filename))[stack_index]

        if self.transform:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.file_list*self.depth)
    