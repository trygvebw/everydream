import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math

class EDValidateBatch(Dataset):
    def __init__(self,
                 data_root,
                 flip_p=0.0,
                 repeats=1,
                 debug_level=0
                 ):

        self.data_root = data_root

        self.image_caption_pairs = dlma(data_root=data_root, debug_level=debug_level).get_all_images()
        
        self.num_images = len(self.image_caption_pairs)

        self._length = max(math.trunc(self.num_images * repeats), 2)

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % len(self.image_caption_pairs)
        example = self.get_image(self.image_caption_pairs[idx])
        return example

    def get_image(self, image_caption_pair):
        example = {}

        image = image_caption_pair[0]

        if not image.mode == "RGB":
            image = image.convert("RGB")

        identifier = image_caption_pair[1]

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["caption"] = identifier

        return example
