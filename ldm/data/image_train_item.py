
import PIL
import numpy as np
from torchvision import transforms
import random
import math

class ImageTrainItem(): # [image, identifier, target_aspect, closest_aspect_wh[w,h], pathname]
    def __init__(self, image: PIL.Image, caption: str, target_wh: list, pathname: str, flip_p=0.0):
        self.caption = caption
        self.target_wh = target_wh
        self.pathname = pathname
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.cropped_img = None

        if image is None:
            self.image = PIL.Image.new(mode='RGB',size=(1,1))
        else:
            self.image = image

    def hydrate(self):
        if type(self.image) is not np.ndarray:
            self.image = PIL.Image.open(self.pathname).convert('RGB')

            cropped_img = self.__autocrop(self.image)

            self.image = cropped_img.resize((512,512), PIL.Image.BICUBIC)

            self.image = self.flip(self.image)
            
            self.image = np.array(self.image).astype(np.uint8)

        self.image = (self.image / 127.5 - 1.0).astype(np.float32)

        return self

    @staticmethod
    def __autocrop(image: PIL.Image, q=.404):
        x, y = image.size

        if x != y:
            if (x>y):
                rand_x = x-y
                rand_y = 0
                sigma = max(rand_x*q,1)
            else:
                rand_x = 0
                rand_y = y-x
                sigma = max(rand_y*q,1)

            if (x>y):
                x_crop_gauss = abs(random.gauss(0, sigma))
                x_crop = min(x_crop_gauss,(x-y)/2)
                x_crop = math.trunc(x_crop)
                y_crop = 0
            else:
                y_crop_gauss = abs(random.gauss(0, sigma))
                x_crop = 0
                y_crop = min(y_crop_gauss,(y-x)/2)
                y_crop = math.trunc(y_crop)
                
            min_xy = min(x, y)
            image = image.crop((x_crop, y_crop, x_crop + min_xy, y_crop + min_xy))
            #print(f"crop: {x_crop} {y_crop}, {x} {y} => {image.size}")

        return image