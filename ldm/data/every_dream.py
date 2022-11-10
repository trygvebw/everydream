import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math
import ldm.data.dl_singleton as dls
from ldm.data.image_train_item import ImageTrainItem

class EveryDreamBatch(Dataset):
    def __init__(self,
                 data_root,
                 repeats=10,
                 flip_p=0.0,
                 debug_level=0,
                 batch_size=1,
                 set='train'
                 ):
        self.data_root = data_root
        self.batch_size = batch_size
        self.debug_level = debug_level
        
        if not dls.shared_dataloader:
            print(" * Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root, debug_level=debug_level, batch_size=self.batch_size, flip_p=flip_p)
        
        self.image_train_items = dls.shared_dataloader.get_all_images()
        
        self.num_images = len(self.image_train_items)

        self._length = math.trunc(self.num_images * repeats)

        print()
        print(f" ** Trainer Set: {set}, steps: {self._length / batch_size:.0f}, num_images: {self.num_images}, batch_size: {self.batch_size}, length w/repeats: {self._length}")
        print()

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        idx = i % self.num_images
        image_train_item = self.image_train_items[idx]
        example = self.__get_image_for_trainer(image_train_item, self.debug_level)
        return example

    @staticmethod
    def __get_image_for_trainer(image_train_item: ImageTrainItem, debug_level=0):
        example = {}

        if debug_level > 1:
            save = True
        image_train_tmp = image_train_item.hydrate(crop=False, save=save)

        example["image"] = image_train_tmp.image
        example["caption"] = image_train_tmp.caption

        return example
