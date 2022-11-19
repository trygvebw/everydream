# script to what cropping does to your images
# execute from root everydream-trainer folder
# ex.
#      (everydream) R:\everydream-trainer>python scripts/test_crop.py
# dumps to /test/output

from  ldm.data.every_dream import EveryDreamBatch
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default=None, help='root path of all your training images, will be recursively searched for images')
parser.add_argument('--resolution', type=int, default=512, help='resolution class, 512, 576, 640, 704, or 768')
args = parser.parse_args()

s = time.perf_counter()

# put in your own data_root here, WARNING don't do this on a lot of images unless you are prepared for it...
if args.data_root is None:
    data_root = "R:/everydream-trainer/test/input"
else:
    data_root = args.data_root

debug_level = 3 # 3 = dump images to disk after cropping and a bunch of crap into the console be warned
batch_size = 1
repeats = 1
crop_jitter = 50
test_cycles = 10
resolution = args.resolution # 512, 576, 640, 704, 768 


every_dream_batch = EveryDreamBatch(data_root=data_root, flip_p=0.0, debug_level=3, \
    batch_size=batch_size, repeats=repeats, crop_jitter=crop_jitter, \
    conditional_dropout=0.1, resolution=resolution, \
    )

for i in range(0,len(every_dream_batch)):
    _ = every_dream_batch[i]

print(f" *TEST* test cycles: {test_cycles}")
elapsed = time.perf_counter() - s
print(f"{__file__} executed in {elapsed:5.2f} seconds.")