import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math
import ldm.data.dl_singleton as dls
from PIL import Image
import gc

class EveryDreamBatch(Dataset):
    def __init__(self,
                 data_root,
                 repeats=10,
                 flip_p=0.0,
                 debug_level=0,
                 batch_size=1,
                 set='train'
                 ):
        #print(f"EveryDreamBatch batch size: {batch_size}")
        self.data_root = data_root
        self.batch_size = batch_size
        self.flip_p = flip_p
        
        if not dls.shared_dataloader:
            print(" * Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root, debug_level=debug_level, batch_size=self.batch_size, flip_p=self.flip_p)
        
        self.image_train_items = dls.shared_dataloader.get_all_images()
        #print(f" * EDB Example {self.image_train_items[0]}")
        
        self.num_images = len(self.image_train_items)

        self._length = math.trunc(self.num_images * repeats)

        print()
        print(f" ** Trainer Set: {set}, steps: {self._length / batch_size:.0f}, num_images: {self.num_images}, batch_size: {self.batch_size}, length w/repeats: {self._length}")
        print()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
        #example = self.get_image(self.image_caption_pairs[idx])
        image_train_item = self.image_train_items[idx]
        #print(f" *** example {example}")

        hydrated_image_train_item = image_train_item.hydrate()

        example = self.get_image_for_trainer(hydrated_image_train_item)
        return example

    def unload_images_over(self, limit):
        print(f" ********** Unloading images over limit {limit}")
        i = 0
        while i < len(self.image_train_items):
            print(self.image_train_items[i])            
            if i > limit:
                self.image_train_items[i][0] = Image.new(mode='RGB', size=(1, 1))
            i += 1
        gc.collect()

    def get_image_for_trainer(self, image_train_item):
        example = {}

        image_train_tmp = image_train_item.as_formatted()

        example["image"] = image_train_tmp.image
        example["caption"] = image_train_tmp.caption

        return example
