import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from ldm.data.data_loader import DataLoaderMultiAspect as dlma
import math
import ldm.data.dl_singleton as dls
class EDValidateBatch(Dataset):
    def __init__(self,
                 data_root,
                 flip_p=0.0,
                 repeats=1,
                 debug_level=0,
                 batch_size=1,
                 set='val',
                 ):

        self.data_root = data_root
        self.batch_size = batch_size

        if not dls.shared_dataloader:
            print("Creating new dataloader singleton")
            dls.shared_dataloader = dlma(data_root=data_root, debug_level=debug_level, batch_size=self.batch_size)
            
        self.image_caption_pairs = dls.shared_dataloader.get_all_images()
        
        self.num_images = len(self.image_caption_pairs)

        self._length = max(math.trunc(self.num_images * repeats), batch_size) - self.num_images % self.batch_size

        print()
        print(f" ** Validation Set: {set}, num_images: {self.num_images}, length: {self._length}, repeats: {repeats}, batch_size: {self.batch_size}, ")
        print(f" ** Validation steps: {self._length / batch_size:.0f}")
        print()

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
