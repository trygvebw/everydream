import os
from PIL import Image
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
        prepared_train_data = self.__prescan_images(debug_level, self.image_paths, flip_p) # ImageTrainItem[]
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level)
        print(f" * DLMA Example {self.image_caption_pairs[0]} images")

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __prescan_images(debug_level: int, image_paths: list, flip_p=0.0):
        decorated_image_train_items = []

        for pathname in image_paths:
            caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]

            txt_file_path = os.path.splitext(pathname)[0] + ".txt"

            if os.path.exists(txt_file_path):
                try:
                    with open(txt_file_path, 'r') as f:
                        print("txt loader")
                        identifier = f.readline().rstrip()
                        if len(identifier) < 1:
                            raise ValueError(f" *** Could not find valid text in: {txt_file_path}")

                except:
                    print(f" *** Error reading {txt_file_path} to get caption, falling back to filename")
                    identifier = caption_from_filename
                    pass
            else:
                identifier = caption_from_filename
            
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            target_wh = min(ASPECTS, key=lambda x:abs(x[0]/x[1]-image_aspect))

            image_train_item = ImageTrainItem(image=None, caption=identifier, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

            decorated_image_train_items.append(image_train_item)

        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0):
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}

        for image_caption_pair in prepared_train_data:
            target_wh = image_caption_pair.target_wh

            if (target_wh[0],target_wh[1]) not in buckets:
                buckets[(target_wh[0],target_wh[1])] = []
            buckets[(target_wh[0],target_wh[1])].append(image_caption_pair) 
        
        print(f" ** Number of buckets: {len(buckets)}")

        if len(buckets) > 1: 
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
        for f in os.listdir(recurse_root):
            current = os.path.join(recurse_root, f)

            # get file ext
            
            if os.path.isfile(current):
                ext = os.path.splitext(f)[1]
                if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                    self.image_paths.append(current)

        sub_dirs = []

        for d in os.listdir(recurse_root):
            current = os.path.join(recurse_root, d)
            if os.path.isdir(current):
                sub_dirs.append(current)

        for dir in sub_dirs:
            self.__recurse_data_root(self=self, recurse_root=dir)
