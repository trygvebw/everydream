import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math

class EveryDreamBatch(Dataset):
    def __init__(self,
                 data_root,
                 repeats=10,
                 flip_p=0.0,
                 debug_level=0,
                 batch_size=1
                 ):
        print(f"EveryDreamBatch batch size: {batch_size}")
        self.data_root = data_root
        self.batch_size = batch_size

        self.image_caption_pairs = dlma(data_root=data_root, debug_level=debug_level, batch_size=self.batch_size).get_all_images()
        
        self.num_images = len(self.image_caption_pairs)

        self._length = math.trunc(self.num_images * repeats)

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        print(f" * Training steps: {self._length / batch_size}")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
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
