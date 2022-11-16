# script to test data loader by itself
# run from training root, edit the data_root manually
# python ldm/data/test_dl.py
from ldm.data.image_train_item import ImageTrainItem
import glob
import os

data_root = "training_samples\multiaspect"

for idx, f in enumerate(glob.iglob(f"{data_root}/*.jpg")):
    for i in range(0, 40):
        #print(f)
        #image: PIL.Image, caption: str, target_wh: list, pathname: str, flip_p=0.0):
        caption = os.path.basename(f)
        caption = os.path.splitext(caption)[0]
        my_iti = ImageTrainItem(None,caption,[512,512],f,0.0)

        my_iti = my_iti.hydrate()

        out_file_path = os.path.join(data_root, "output", f"{caption}_{i}.jpg")
        #print(out_file_path)
        my_iti.cropped_img.save(out_file_path)


