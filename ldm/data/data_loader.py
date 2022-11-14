import os
from PIL import Image
import random
from ldm.data.image_train_item import ImageTrainItem

HUGE_ASPECTS = [[640,640], # 409600 1:1 
    [704,576],[576,704], # 405504 1:1.25
    [768,512],[512,768], # 393216 1:1.5
    [896,448],[448,896], # 401408 1:2
    [1024,384],[384,1024], # 393216 1:2.667
    [1280,320],[320,1280], # 409600 1:4
    [1408,256],[256,1408], # 360448 1:5.5
    [1472,256],[256,1472], # 376832 1:5.75
    [1536,256],[256,1536], # 393216 1:6
    [1600,256],[256,1600], # 409600 1:6.25
]

BIG_ASPECTS = [[576,576], # 331776 1:1\
    [640,512],[512,640], # 327680 1.25:1\
    [704,448],[448,704], # 314928 1.5625:1
    [832,384],[384,832], # 317440 2.1667:1\
    [1024,320],[320,1024], # 327680 3.2:1\
    [1280,256],[256,1280], # 327680 5:1\
]

ASPECTS = [[512,512], # 1 262144\
        [576,448],[448,576], # 1.29 258048\
        [640,384],[384,640], # 1.67 245760\
        [704,384],[384,704], # 1.83 245760\
        [768,320],[320,768], # 2.4 245760\
        [832,256],[256,832], # 3.25 212992\
        [896,256],[256,896], # 3.5 229376\
        [960,256],[256,960],  # 3.75 245760\
        [1024,256],[256,1024],  # 4 245760\
    ]
        
class DataLoaderMultiAspect():
    """
    Data loader for multi-aspect-ratio training and bucketing

    data_root: root folder of training data
    batch_size: number of images per batch
    flip_p: probability of flipping image horizontally (i.e. 0-0.5)
    """
    def __init__(self, data_root, seed=555, debug_level=0, batch_size=1, flip_p=0.0, big_mode=0):
        self.image_paths = []
        self.debug_level = debug_level
        self.flip_p = flip_p
        self.big_mode = big_mode

        print(" Preloading images...")

        self.__recurse_data_root(self=self, recurse_root=data_root)
        random.Random(seed).shuffle(self.image_paths)
        prepared_train_data = self.__prescan_images(debug_level, self.image_paths, flip_p) # ImageTrainItem[]
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level)

        if debug_level > 0: print(f" * DLMA Example: {self.image_caption_pairs[0]} images")

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __read_caption_from_file(self, file_path, fallback_caption):
        caption = fallback_caption
        try:
            with open(file_path, 'r') as caption_file:
                caption = caption_file.read()
        except:
            print(f" *** Error reading {file_path} to get caption, falling back to filename")
            caption = fallback_caption
            pass
        return caption

    def __prescan_images(self, debug_level: int, image_paths: list, flip_p=0.0):
        """
        Create ImageTrainItem objects with metadata for hydration later 
        """
        decorated_image_train_items = []

        for pathname in image_paths:
            caption_from_filename = os.path.splitext(os.path.basename(pathname))[0].split("_")[0]

            txt_file_path = os.path.splitext(pathname)[0] + ".txt"
            caption_file_path = os.path.splitext(pathname)[0] + ".caption"

            if os.path.exists(txt_file_path):
                caption = self.__read_caption_from_file(txt_file_path, caption_from_filename)                
            elif os.path.exists(caption_file_path):
                caption = self.__read_caption_from_file(caption_file_path, caption_from_filename)                
            else:
                caption = caption_from_filename

            if debug_level > 1: print(f" * DLMA file: {pathname} with caption: {caption}")
            
            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            aspects = [ASPECTS, BIG_ASPECTS, HUGE_ASPECTS][self.big_mode]

            target_wh = min(aspects, key=lambda x:abs(x[0]/x[1]-image_aspect))

            image_train_item = ImageTrainItem(image=None, caption=caption, target_wh=target_wh, pathname=pathname, flip_p=flip_p)

            decorated_image_train_items.append(image_train_item)

        return decorated_image_train_items

    @staticmethod
    def __bucketize_images(prepared_train_data: list, batch_size=1, debug_level=0):
        """
        Put images into buckets based on aspect ratio with batch_size*n images per bucket, discards remainder
        """
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
