import numpy as np
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
                 debug_level=0,
                 batch_size=1
                 ):
        print(f"EDValidateBatch batch size: {self.batch_size}") if debug_level > 0 else None
        self.data_root = data_root
        self.batch_size = batch_size

        self.image_caption_pairs = dlma(data_root=data_root, debug_level=debug_level, batch_size=self.batch_size).get_all_images()

        # most_subscribed_aspect_ratio = self.most_subscribed_aspect_ratio()
        # self.image_caption_pairs = [image_caption_pair for image_caption_pair in self.image_caption_pairs if image_caption_pair[0].size == aspect_ratio]
        
        self.num_images = len(self.image_caption_pairs)

        self._length = max(math.trunc(self.num_images * repeats), 1)

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % len(self.image_caption_pairs)
        example = self.get_image(self.image_caption_pairs[idx])
        #print caption and image size
        print(f"Caption: {example['image'].shape} {example['caption']}") 
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

    def filter_aspect_ratio(self, aspect_ratio):
        # filter the images to only include the given aspect ratio
        self.image_caption_pairs = [image_caption_pair for image_caption_pair in self.image_caption_pairs if image_caption_pair[0].size == aspect_ratio]
        self.num_images = len(self.image_caption_pairs)
        self._length = max(math.trunc(self.num_images * self.repeats), 2)

    def most_subscribed_aspect_ratio(self):
        # find the image size with the highest number of images
        aspect_ratios = {}
        for image_caption_pair in self.image_caption_pairs:
            image = image_caption_pair[0]
            aspect_ratio = image.size
            if aspect_ratio in aspect_ratios:
                aspect_ratios[aspect_ratio] += 1
            else:
                aspect_ratios[aspect_ratio] = 1

        return max(aspect_ratios, key=aspect_ratios.get)
