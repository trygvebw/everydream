
from PIL import Image
import numpy as np
from torchvision import transforms

class ImageTrainItem(): # [image, identifier, target_aspect, closest_aspect_wh[w,h], pathname]
    def __init__(self, image: Image, caption: str, target_wh: list, pathname: str, flip_p=0.0):
        self.caption = caption
        self.target_wh = target_wh
        #self.target_aspect = target_aspect
        self.pathname = pathname
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        if image is None:
            self.image = Image.new(mode='RGB',size=(1,1))
        else:
            self.image = image
        #image_train_item.image = image.resize((image_train_item.closest_aspect_wh[0], image_train_item.closest_aspect_wh[1]), Image.BICUBIC)

    def hydrate(self):
        self.image = self.image.resize(self.target_wh, Image.BICUBIC)
        
        if not self.image.mode == "RGB":
            self.image = self.image.convert("RGB")

        self.image = self.flip(self.image)
        self.image = np.array(self.image).astype(np.uint8)

        self.image = (self.image / 127.5 - 1.0).astype(np.float32)

        return self