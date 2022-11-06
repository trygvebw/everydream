# script to test data loader by itself
# run from training root, edit the data_root manually
# python ldm/data/test_dl.py
import every_dream
import time


s = time.perf_counter()

data_root = "r:/everydream-trainer/training_samples/ff7r"

batch_size = 1
every_dream_batch = every_dream.EveryDreamBatch(data_root=data_root, flip_p=0.0, debug_level=0, batch_size=batch_size, repeats=1)

print(f" *TEST*  batch type: {type(every_dream_batch)}")
i = 0
is_next = True
curr_batch = []

while is_next and i < 30 and i < len(every_dream_batch):
    try:
        example = every_dream_batch[i]
        if example is not None:
            #print(f"example type: {type(example)}") # dict
            #print(f"example keys: {example.keys()}") # dict_keys(['image', 'caption'])
            #print(f"example image type: {type(example['image'])}") # numpy.ndarray
            if i%batch_size == 0:
                curr_batch = example['image'].shape
            img_in_right_batch = curr_batch == example['image'].shape
            print(f" *TEST*  example image shape: {example['image'].shape} {i%batch_size} {img_in_right_batch}")
            print(f" *TEST*  example caption: {example['caption']}")

            if not img_in_right_batch:
                raise Exception("Current image in wrong batch")
            #print(f"example caption: {example['caption']}") # str
        else:
            is_next = False
        i += 1
    except IndexError:
        is_next = False
        print(f"IndexError: {i}")
        pass
    # for idx, batches in every_dream_batch:
    # print(f"inner example type: {type(batches)}")
    # print(type(batches))
    # print(type(batches[0]))
    # print(dir(batches))
    #h, w = batches.image.size
    #print(f"{idx:05d}-{idx%6:02d}EveryDreamBatch image caption pair: w:{w} h:{h} {batches.caption[1]}")
print(f" *TEST* test cycles: {i}")
print(f" *TEST* EveryDreamBatch epoch image length: {len(every_dream_batch)}")
elapsed = time.perf_counter() - s
print(f"{__file__} executed in {elapsed:5.2f} seconds.")