# script to test data loader by itself
# run from training root, edit the data_root manually
# python ldm/data/test_dl.py
import ldm.data.data_loader as dl

data_root = "r:/everydream-trainer/test/input"

data_loader = dl.DataLoaderMultiAspect(data_root=data_root, batch_size=2, seed=555, debug_level=2)

image_caption_pairs = data_loader.get_all_images()

print(f"Loaded {len(image_caption_pairs)} image-caption pairs")

for image_caption_pair in image_caption_pairs:
    print(image_caption_pair)
    

print(f"**** Done loading. Loaded {len(image_caption_pairs)} images from data_root: {data_root} ****")