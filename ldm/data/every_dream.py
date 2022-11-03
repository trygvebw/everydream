import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from ldm.data.data_loader import DataLoader as dl

class EveryDreamBatch(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=10,
                 interpolation="bicubic",
                 flip_p=0.0,
                 center_crop=False
                 ):

        self.data_root = data_root

        self.image_paths = dl(data_root=data_root, quiet=False).get_all_images()
        
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.center_crop = center_crop

        self._length = self.num_images * repeats

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % len(self.image_paths)
        example = self.get_image(self.image_paths[idx])
        return example

    def get_image(self, image_path):
        example = {}

        image = Image.open(image_path)

        if not image.mode == "RGB":
            image = image.convert("RGB")

        pathname = Path(image_path).name

        parts = pathname.split("_")
        identifier = parts[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            h, w, = img.shape[0], img.shape[1]
            img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size),
                                 resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        example["caption"] = identifier

        return example
