import os
from PIL import Image

ASPECTS = [[512,512], # 1 262144\
        [576,448],[448,576], # 1.29 258048\
        [640,384],[384,640], # 1.67 245760\
        [768,320],[320,768], # 2.4 245760\
        [832,256],[256,832], # 3.25 212992\
        [896,256],[256,896], # 3.5 229376\
        [960,256],[256,960],  # 3.75 245760\
        [1024,256],[256,1024]  # 4 245760\
    ]

class DataLoader():
    def __init__(self, data_root, seed=555, debug_level=0):
        self.data_root = data_root
        self.image_paths = []
        self.image_caption_pairs = []
        self.seed = 555
        self.debug_level = debug_level

        if self.debug_level > 0:
            print(f"**** Loading data_root: {data_root} ****") 
            print(f" shuffle seed:{self.seed}")

        self.__recurse_images(self=self, recurse_root=data_root)
        
        import random
        random.Random(self.seed).shuffle(self.image_paths)
        self.seed = self.seed + 1
        
        print(f"**** Done loading. Loaded {len(self.image_paths)} images from data_root: {self.data_root} ****") if self.debug_level > 0 else None
        print(self.image_paths) if self.debug_level > 1 else None

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __recurse_images(self, recurse_root):
        i = 0
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)
            if os.path.isfile(current):
                i += 1
                self.image_paths.append(current)

        print(f"  ** Found {str(i).rjust(5,' ')} files in", recurse_root) if self.debug_level > 0 else None

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_images(self=self, recurse_root=dir)

class DataLoaderMultiAspect(DataLoader):
    def __init__(self, data_root, seed=555, debug_level=0):
        super().__init__(data_root, seed, debug_level)

        self.image_caption_pairs = self.__load_images(self, self.image_paths)

    @staticmethod
    def __load_images(self, image_paths):
        decorated_image_paths = []
        print("* Loading images using multi-aspect-ratio loader *") if self.debug_level > 0 else None
        i = 0
        for pathname in image_paths:
            print(pathname) if self.debug_level > 1 else None
            parts = os.path.basename(pathname).split("_")
            identifier = parts[0]

            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            closest_aspect = min(ASPECTS, key=lambda x:abs(x[0]/x[1]-image_aspect))
            print(f"{closest_aspect} {closest_aspect[0]/closest_aspect[1]}") if self.debug_level > 1 else None

            target_aspect = closest_aspect[0]/closest_aspect[1]
            #crop image to closest aspect
            if closest_aspect[0] == closest_aspect[1]:
                pass
            if target_aspect < image_aspect:
                crop_width = (width - (height * closest_aspect[0] / closest_aspect[1])) / 2
                print(f"  ** Cropping width: {crop_width}") if self.debug_level > 1 else None
                image = image.crop((crop_width, 0, width - crop_width, height))
            else:
                crop_height = (height - (width * closest_aspect[1] / closest_aspect[0])) / 2
                print(f"  ** Cropping height: {crop_height}") if self.debug_level > 1 else None
                image = image.crop((0, crop_height, width, height - crop_height))
 
            image = image.resize((closest_aspect[0], closest_aspect[1]), Image.BICUBIC)
            print(f"  ** {image_aspect} resized to {closest_aspect[0]/closest_aspect[1]}:{closest_aspect}") if self.debug_level > 1 else None

            if self.debug_level > 1:
                print(f"  ** Multiaspect debug: saving resized image to outputs/{i}.png")
                image.save(f"outputs/{i}.png",format="png") 
            i += 1
            
            decorated_image_paths.append([image, identifier])

        return decorated_image_paths

    def get_all_image_caption_pairs(self):
        self.image_caption_pairs = self.__load_images(self, self.image_paths)
       
        return self.image_paths
