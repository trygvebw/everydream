import os
from PIL import Image
import gc

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
    def __init__(self, data_root, seed=555, debug_level=0, batch_size=2):
        self.image_paths = []
        self.debug_level = debug_level

        self.__recurse_images(self=self, recurse_root=data_root)

        prepared_train_data = self.__crop_resize_images(debug_level, self.image_paths)
        #print(f"prepared data __init__: {prepared_train_data}")
        self.image_caption_pairs = self.__bucketize_images(prepared_train_data, batch_size=batch_size, debug_level=debug_level)

        print(f"**** Done loading. Loaded {len(self.image_paths)} images from data_root: {data_root} ****") if self.debug_level > 0 else None
        print(self.image_paths) if self.debug_level > 1 else None

        gc.collect()

    def get_all_images(self):
        return self.image_caption_pairs

    @staticmethod
    def __crop_resize_images(debug_level, image_paths):
        decorated_image_paths = []
        print("* Loading images using multi-aspect-ratio loader *") if debug_level > 1 else None
        i = 0
        for pathname in image_paths:
            print(pathname) if debug_level > 1 else None
            parts = os.path.basename(pathname).split("_")
            parts[-1] = parts[-1].split(".")[0]
            identifier = parts[0]

            image = Image.open(pathname)
            width, height = image.size
            image_aspect = width / height

            closest_aspect = min(ASPECTS, key=lambda x:abs(x[0]/x[1]-image_aspect))

            target_aspect = closest_aspect[0]/closest_aspect[1]

            if closest_aspect[0] == closest_aspect[1]:
                pass
            if target_aspect < image_aspect:
                crop_width = (width - (height * closest_aspect[0] / closest_aspect[1])) / 2
                #print(f"  ** Cropping width: {crop_width}") if debug_level > 1 else None
                image = image.crop((crop_width, 0, width - crop_width, height))
            else:
                crop_height = (height - (width * closest_aspect[1] / closest_aspect[0])) / 2
                #print(f"  ** Cropping height: {crop_height}") if debug_level > 1 else None
                image = image.crop((0, crop_height, width, height - crop_height))
 
            image = image.resize((closest_aspect[0], closest_aspect[1]), Image.BICUBIC)

            if debug_level > 1:
                print(f"  ** Multi-aspect debug: saving resized image to outputs/{i}.png")
                image.save(f"outputs/{i}.png",format="png") 
            i += 1
            
            decorated_image_paths.append([image, identifier])

        return decorated_image_paths

    @staticmethod
    def __bucketize_images(prepared_train_data, batch_size=1, debug_level=0):
        # TODO: this is not terribly efficient but at least linear time
        buckets = {}

        for image_caption_pair in prepared_train_data:
            image = image_caption_pair[0]
            width, height = image.size
            if (width, height) not in buckets:
                buckets[(width, height)] = []
            buckets[(width, height)].append(image_caption_pair)
        
        for bucket in buckets:
            truncated_count = len(buckets[bucket]) % batch_size
            current_bucket_size = len(buckets[bucket])
            buckets[bucket] = buckets[bucket][:current_bucket_size - truncated_count]
            print(f"  ** Bucket {bucket} with {current_bucket_size} will truncate {truncated_count} images due to batch size {batch_size}") if debug_level > 0 else None

        image_caption_pairs = []
        for bucket in buckets:
            image_caption_pairs.extend(buckets[bucket])
        
        return image_caption_pairs

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