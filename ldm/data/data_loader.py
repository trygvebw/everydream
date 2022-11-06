import os
from PIL import Image
import gc
import random
from ldm.data.image_train_item import ImageTrainItem

ASPECTS = [[512,512], # 1 262144\
        [576,448],[448,576], # 1.29 258048\
        [640,384],[384,640], # 1.67 245760\
        [768,320],[320,768], # 2.4 245760\
        [832,256],[256,832], # 3.25 212992\
        [896,256],[256,896], # 3.5 229376\
        [960,256],[256,960],  # 3.75 245760\
        [1024,256],[256,1024]  # 4 245760\
    ]
        
class DataLoaderMultiAspect():
    def __init__(self, data_root, seed=555, debug_level=0, batch_size=1, flip_p=0.0):
        self.image_paths = []
        self.debug_level = debug_level
        self.flip_p = flip_p

        print(" Preloading images...")

        self.__recurse_data_root(self=self, recurse_root=data_root)
        random.Random(seed).shuffle(self.image_paths)
        prepared_train_data = self.__prescan_images(debug_level, self.image_paths, flip_p)
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level)
        print(f" * DLMA Example {self.image_caption_pairs[0]} images")

        gc.collect()

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __prescan_images(debug_level: int, image_paths: list, flip_p=0.0):
        decorated_image_train_items = []

        for pathname in image_paths:
            parts = os.path.basename(pathname).split("_")
            
            # untested
            # txt_filename = parts[0] + ".txt"
            # if os.path.exists(txt_filename):
            #     try:
            #         with open(txt_filename, 'r') as f:
            #             identifier = f.read()
            #             identifier.rstrip()
            #     except:
            #         print(f" *** Error reading {txt_filename} to get caption")
            #         identifier = parts[0]
            #         pass
            # else:
            #     identifier = parts[0]

            identifier = parts[0]
            
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(ASPECTS, key=lambda x:abs(x[0]/x[1]-image_aspect))

            image_train_item = ImageTrainItem(image=None, caption=identifier, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

            # put placeholder image in the list and return meta data
            decorated_image_train_items.append(image_train_item)
        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data, batch_size=1, debug_level=0):
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}

        for image_caption_pair in prepared_train_data:
            image = image_caption_pair.image
            width, height = image.size

            if (width, height) not in buckets:
                buckets[(width, height)] = []
            buckets[(width, height)].append(image_caption_pair) # [image, identifier, target_aspect, closest_aspect_wh[w,h], pathname]
        
        print(f" ** Number of buckets: {len(buckets)}")

        if len(buckets) > 1: # don't bother truncating if everything is the same aspect ratio
            for bucket in buckets:
                truncate_count = len(buckets[bucket]) % batch_size
                current_bucket_size = len(buckets[bucket])
                buckets[bucket] = buckets[bucket][:current_bucket_size - truncate_count]
                print(f"  ** Bucket {bucket} with {current_bucket_size} will drop {truncate_count} images due to batch size {batch_size}") if debug_level > 0 else None

        # flatten the buckets
        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])
        
        return image_caption_pairs

    @staticmethod
    def __recurse_data_root(self, recurse_root):
        i = 0

        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)
            # get file ext
            
            if os.path.isfile(current):
                ext = os.path.splitext(f)[1]
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    i += 1
                    self.image_paths.append(current)

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_data_root(self=self, recurse_root=dir)

    # @staticmethod
    # def hydrate_image(self, image_path, target_aspect, closest_aspect_wh):
    #     image = Image.open(example[4]) # 5 is the path
    #     print(image)
    #     width, height = image.size
    #     image_aspect = width / height
    #     target_aspect = width / height

    #     if example[3][0] == example[3][1]:
    #         pass
    #     if target_aspect < image_aspect:
    #         crop_width = (width - (width * example[3][0] / example[3][1])) / 2
    #         image = image.crop((crop_width, 0, width - crop_width, height))
    #     else:
    #         crop_height = (height - (width * example[3][1] / example[3][0])) / 2
    #         image = image.crop((0, crop_height, width, height - crop_height))

    #     example[0] = image.resize((example[3][0], example[3][1]), Image.BICUBIC)
    #     return example