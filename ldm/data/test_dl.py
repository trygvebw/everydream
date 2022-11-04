# script to test data loader
# python ldm/data/test_dl.py
import data_loader

data_root = "r:/everydream-trainer/training_samples/multiaspect"

data_loader = data_loader.DataLoaderMultiAspect(data_root=data_root, seed=555, debug_level=2)

image_caption_pairs = data_loader.get_all_images()

print(f"Loaded {len(image_caption_pairs)} image-caption pairs")

for image_caption_pair in image_caption_pairs:
    print(image_caption_pair)
    print(image_caption_pair[1])

print(f"**** Done loading. Loaded {len(image_caption_pairs)} images from data_root: {data_root} ****")