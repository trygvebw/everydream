
import PIL
import numpy as np
from torchvision import transforms

class ImageTrainItem(): # [image, identifier, target_aspect, closest_aspect_wh[w,h], pathname]
    def __init__(self, image: PIL.Image, caption: str, target_wh: list, pathname: str, flip_p=0.0):
        self.caption = caption
        self.target_wh = target_wh
        self.pathname = pathname
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

        if image is None:
            self.image = PIL.Image.new(mode='RGB',size=(1,1))
        else:
            self.image = image

    def hydrate(self):
        if type(self.image) is not np.ndarray:
            self.image = PIL.Image.open(self.pathname).convert('RGB')

            self.image = self.image.resize((self.target_wh), PIL.Image.BICUBIC)

            self.image = self.flip(self.image)
            self.image = np.array(self.image).astype(np.uint8)

        self.image = (self.image / 127.5 - 1.0).astype(np.float32)

        return self