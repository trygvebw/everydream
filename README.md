# Every Dream trainer for Stable Diffusion

This is a bit of a divergence from other fine tuning methods out there for Stable Diffusion.  No more "DreamBooth" stuff like tokens, classes, or regularization, though I thank the DreamBooth training community for sharing information and techniques.  Yet, it is time to move on to explore more capability in fine tuning.

## Onward to Every Dream
This trainer is focused on enabling fine tuning with new training data plus weaving in original, ground truth images scraped from the web via Laion dataset or other publically available ML image sets.  Compared to DreamBooth, concepts such as regularization have been removed in favor of adding back ground truth data (ex. Laion), and token/class concepts are removed and replaced by per-image captioning for training, more or less equal to how Stable Diffusion was trained itself. This is a shift back to the original training code and methodology for fine tuning for general cases.

To get the most out of this trainer, you will need to curate a data set to be trained in addition to collect ground truth images to help preserve the model integrity and character.  Luckily, there are additional tools below to help enable that, and will grow over time.

## Techniques

This is a general purpose fine tuning app.  You can train large or small scale with it and everything in between.

Check out [MICROMODELS.MD](./MICROMODELS.MD) for a quickstatr guide and example for quick model creation with a small data set.  It is suited for training one or two subects with 20-50 images with no preservation in 10-25 minutes.

Or [README-FF7R.MD](./README-FF7R.MD) for large scale training of many characters with model preservation.

More info coming soon on even larger training.

## Image Captioning

This trainer is built to use the filenames of your images as "captions" on a per-image basis, *so the entire Latent Diffusion model can be trained effectively.*  **Image captioning is a big step forward.** 

### Formatting

The filenames are using for captioning, with a split on underscore so you can have "duplicate" captioned images.  Examples of valid filenames:

    a photo of John Jacob Jingleheimerschmidt riding a bicycle.webp
    a pencil drawing of john jacob jingleheimerscmidt.jpg
    john jacob jingleheimerschmidt sitting on a bench in a park with trees in the background_(1).png
    john jacob jingleheimerschmidt sitting on a bench in a park with trees in the background_(2).png

In the 3rd and 4th example above, the _(1) and _(2) are ignored and not considered by the trainer.  This is useful if you end up with duplicate filenames but different image contents for whatever reason. 

### Data set organization

You will need to organize your files into datasets with a single layer of subfolders.  

While you can simply stuff everything, new training and ground truth data all in one subfolder, for organization purposes I suggest splitting up your subfolders in a fashion such as the following:

    /training_samples/MyProject
    /training_samples/MyProject/man
    /training_samples/MyProject/man_laion
    /training_samples/MyProject/man_nvflickr
    /training_samples/MyProject/paintings_laion
    /training_samples/MyProject/drawings_laion

In the above example, "/training_samples/MyProject" will be your root folder for the command line.  It must be devoid of anything but the subfolders.  **The subfolders again are purely for your own organizational purposes, the names of the subfolders do not matter to the trainer.** It's up to you how you want to name or organize subfolders, the only requirement is that you use a single layer of subfolders and the root folder for your project contains nothing but the subfolders.  You must not put images or other files directly into /training_samples/MyProject. 

Also in the above example, /training_samples/MyProject/man would contain new training images you want to "teach" the model, and the man_laion and man_nvflickr sets would contain images scraped from laion or other original sources (see below for possible sources). It's up to you what you want to include.

As you build your data set, you may find it is easiest to organize in this way to track your balance between new training data and ground truth used to preserve the model integrity.  For instance, if you have 500 new training images in ../man you may with to use 500  in the /man_laion and another 500 in /man_nvflickr.  You can then experiment by removing different folders to see the effects on training quality.

### Suggestions

The more data you add from ground truth data sets such as Laion, the more training you will get away with without "damaging" the original model.  The wider variety of data in the ground truth portion of your dataset, the less likely your training images are to "bleed" into the rest of your model, losing qualities like the ability to generate images of other styles you are not training.  This is about knowledge retention in the model by refeeding it the same data it was originally trained on.

## Ground truth data sources and data engineering

Visit [EveryDream Data Engineering Tools](https://github.com/victorchall/EveryDream) to find a web scraper that can pull down images from the Laion dataset.  You should consider that your first step before using this trainer to collect data.

I suggest pulling down all the files for this set in particular: [https://huggingface.co/datasets/laion/laion2B-en-aesthetic](https://huggingface.co/datasets/laion/laion2B-en-aesthetic) to use with the web scraper.

There is information is in the EveryDream Data Engineering Tools link above on how to run the web scrape.  The webscrape takes zero GPU power, so you can run it locally on any PC with Python before renting GPU power if needed.  If you are interested in moving to larger scope projects I recommend investing time to curae your data sets as they can be reused. 

The Nvidia Flickr set is also helpful and in a fairly "ready to use" format besides renaming the files: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 
For this trainer, I suggest "close up photo of a person" for captioning of this dataset. If you want, you can go further and separate male/female photos and caption them "close up phot of a man" or "..a woman" as you see fit.  "a close up of a person" is also acceptable, dropping "photo".  You can simply select all in windows, F2 to rename and type "a close up of a person_" **without the quotes but with the underscore** to format the filename captions in a way this trainer can use.
## Starting training

An example comand to start training:

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t  --actual_resume sd_v1-5_vae.ckpt -n MyProjectName --gpus 0, --data_root training_samples\MyProject

In the above, the source training data is expected to be laid out in subfolders of training_samples\MyProject as described in above sections. It will use the first Nvidia GPU in the system, and resume from the checkpoint named "sd_v1-5_vae.ckpt".  "-n MyProjectName" is merely a name for the folder where logs will be written during training, which appear under /logs. 
## Testing

I strongly recommend attempting to undertrain via the repeats and max_epochs above to test your model before continuing.  Try one epoch at 10, then grab the ckpt file from the log folder.  The ckpt will be dumped to a folder such as \logs\MyPrject2022-10-25T20-37-40_MyProject date stamped to the start of training. There are also test images in the \logs\images\train folder that spit out periodically based on another finetune yaml setting:

      callbacks:
        image_logger:
        target: main.ImageLogger
        params:
            batch_frequency: 300

The images will often not all be fully formed, and are randomly selected based on the last few training images, but it's a good idea to start learning what to watch for in those images. 

To continue training on a checkpoint, grab the last ckpt file \logs\MyPrject2022-10-25T20-37-40_MyProject\checkpoints and move it back to your base folder and just change the --actual_resume pointer to last.ckpt such as the following:

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t  --actual_resume last.ckpt -n MyProjectName --gpus 0, --data_root training_samples\MyProject

Again, good idea to think about adjusting your repeats before continuing!  

## Finetune yaml adjustments

Depending on your project, a few settings may be useful to tweak or adjust.  In [Starting Training](#starting_training) I'm using __v1-finetune_everydream.yaml__ here but you can make your own copies if you like with different adjustments and save them for your projects.  It is a good idea to get familar with this file as tweaking can be useful as you train.

I'll highlight the following settings at the end of the file:

    trainer:
      benchmark: True
      max_epochs: 2
      max_steps: 99000

max_epochs will halt training.  I suggest ending on a clean end of an epoch rather than using a steps limit, so defaults are configured as such.  2 epochs is not a lot but it is a good point to check before continuing as you can always continue training but you can't go back if you overdo it. 

      train:
        target: ldm.data.every_dream.EveryDreamBatch
        params:
            size: 512
            set: train
            repeats: 20

Above, the repeats defines the number of times each training image is trained on per epoch.  This is mainly a control to balance against validation.  For large scale training with 100+ images per subject you may find just 20 repeats with 1 epoch or 10 repeats with 2 epochs is a good place to stop and check your outputs by loading your file into an inference repo.

The only difference between 20 repeats 1 epoch and 10 repeats 2 epochs is the later will run validation twice (always once per epoch), which costs some extra steps and time.  Once you develop a "feel" for your projects you may adjust increase repeats on your first training off a base model to save a bit of time on the validation steps, test, then continue.  You may which to, for example, doing 1 epoch at 20 repeats, check, then do one more epoch at 5 repeats if you feel it is "close" to done. 

The above settings are a good place at least for humanoid subjects with 100+ images per subject, though some users may find less humanoid subjects require more training, such as cartoons, creatures, etc.

You are also free to move data in and out of your training_samples/MyProject folder between training sessions.  If you have multiple subjects and your number of images between them is a bit mismatched in number, say, 100 for one and only 60 for another, you can try running one epoch 25 repeats, then remove the character with 100 images and train just the one with the 60 images for another epoch at 5 repeats.  It's best to try to keep the data evenly spread, but sometimes that is diffcult.  You may also find certain characters are harder to train, and need more on their own.  Again, test!  Go generate images between 

    data:
      target: main.DataModuleFromConfig
      params:
        batch_size: 6
        num_workers: 8

Batch size determine how many images are loaded and trained on in parallel. 6 will work on a 24GB GPU, 1 will only reduce VRAM use to about 20GB.  This will divide the number of steps used as well, but one epoch is still "repeats" number of trainings on each image.  

I recommend not worrying about step count, but you can calcuate it per epoch as repeats * number_of_training_images / batch_size * (1+1/repeats).  For example, 500 training images with 10 repeats and batch size of six will perform 835 steps per epoch.

### Additional notes

Thanks go to the CompVis team for the original training code, Xaiver Xiao for the DreamBooth implementation and tweaking of trainer configs to stuff it into a 24GB card, and Kane Wallmann for code take image captions from the filenames.

References:

[Compvis Stable Diffusion](https://github.com/CompVis/stable-diffusion)

[Xaiver Xiao's DreamBooth implementation](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)

[Kane Wallmann's captioning capability](https://github.com/kanewallmann/Dreambooth-Stable-Diffusion)