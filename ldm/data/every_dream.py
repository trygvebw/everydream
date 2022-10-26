import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path

class EveryDreamBatch(Dataset):
    def __init__(self,
                 data_root,
                 size=None,
                 repeats=10,
                 interpolation="bicubic",
                 flip_p=0.0,
                 set="train",
                 center_crop=False,
                 reg=False
                 ):

        self.data_root = data_root
        self.reg = reg
        self.image_paths = []

        classes = os.listdir(self.data_root)
        print(f"**** Loading data set: data_root: {data_root}, as set: {set}")

        for cl in classes:
            class_path = os.path.join(self.data_root, cl)
            for file_path in os.listdir(class_path):
                image_path = os.path.join(class_path, file_path)
                self.image_paths.append(image_path)

        import random
        # improve multi-class training by mixing order of training set, avoid training on one class N times in a row
        # if trainer crashes between epochs and you resume at least it isn't heavily biasing early files in dir order
        self.image_paths = random.Random(555).shuffle(self.image_paths)

        # self._length = len(self.image_paths)
        self.num_images = len(self.image_paths)
        self._length = self.num_images

        self.center_crop = center_crop

        if set == "train":
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

        example["caption"] = identifier

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
        return example
