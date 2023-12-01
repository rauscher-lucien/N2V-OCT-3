import torch
import numpy as np
import copy


class ToNumpyArray:
    def __call__(self, data):
        # Transform the data into a numpy array
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except Exception as e:
                raise ValueError(f"Unable to convert input to a NumPy array: {e}")
        # Convert the data to float32
        data = data.astype(np.float32)

        return data


class NormalizeArray:
    def __call__(self, data):
        # Step 1: Find the minimum and maximum values
        min_value = np.min(data)
        max_value = np.max(data)

        # Step 2: Normalize the image to the range [0, 1]
        data = (data - min_value) / (max_value - min_value)

        return data


class RandomFlip:
    def __call__(self, data):

        if np.random.rand() > 0.5:
            data = np.fliplr(data)

        if np.random.rand() > 0.5:
            data = np.flipud(data)

        return data


class RandomCrop:
    def __init__(self, patch_size=50):

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)  # Convert integer to (height, width)
        elif isinstance(patch_size, tuple) and len(patch_size) == 2:
            self.patch_size = patch_size  # Use the provided (height, width) tuple
        else:
            raise ValueError("Invalid patch_size. Should be an integer or a tuple (height, width).")

        self.new_h, self.new_w = self.patch_size

    def __call__(self, data):
        h, w = data.shape
        new_h, new_w = self.patch_size

        if new_w >= w or new_h >= h:
            raise ValueError("Output size should be smaller than the input array.")

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        return data[top: top + new_h, left: left + new_w]


class GenerateN2VMask:
    def __init__(self, percentage_of_pixels=0.198, window=(5, 5)):

        self.percentage_of_pixels = percentage_of_pixels
        self.window_h, self.window_w = window

    def __call__(self, data):

        h, w = data.shape
        pixel_num = int(h * w * self.percentage_of_pixels)

        window_h = self.window_h
        window_w = self.window_w

        mask = np.ones(data.shape)
        label = copy.deepcopy(data)
        input = label

        idy_msk = np.random.randint(0, h, pixel_num)
        idx_msk = np.random.randint(0, w, pixel_num)

        idy_neigh = np.random.randint(window_h // 2, window_h // 2 + 1, pixel_num)
        idx_neigh = np.random.randint(window_w // 2, window_w // 2 + 1, pixel_num)

        idy_msk_neigh = idy_msk + idy_neigh
        idx_msk_neigh = idx_msk + idx_neigh

        idy_msk_neigh = np.clip(idy_msk_neigh, 0, h - 1)
        idx_msk_neigh = np.clip(idx_msk_neigh, 0, w - 1)

        id_msk = (idy_msk, idx_msk)
        id_msk_neigh = (idy_msk_neigh, idx_msk_neigh)

        input[id_msk] = label[id_msk_neigh]
        mask[id_msk] = 0.0

        return {'input': input[..., None], 'label': label[..., None], 'mask': mask[..., None]}

class ToTensor:
    def __call__(self, data):
        if isinstance(data, dict):
            # Convert each value (assumed to be a numpy array or tensor) in the dictionary
            # to a PyTorch tensor and rearrange the axes
            return {key: torch.from_numpy(np.moveaxis(value.copy(), [0, 1, 2], [1, 2, 0])) 
                    if isinstance(value, np.ndarray) else value for key, value in data.items()}
        else:
            raise ValueError("Input must be a dictionary of tensors.")



class BackToNumpyArray:
    def __call__(self, data):
        if isinstance(data, torch.Tensor):
            # Convert the input tensor to a NumPy array and rearrange the axes
            numpy_array = data.detach().cpu().numpy()
            return numpy_array
        else:
            raise ValueError("Input must be a tensor.")



    

