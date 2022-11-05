# Every Dream trainer for Stable Diffusion

This is a bit of a divergence from other fine tuning methods out there for Stable Diffusion.  No more "DreamBooth" stuff like tokens, classes, or regularization, though I thank the DreamBooth training community for sharing information and techniques.  Yet, it is time to move on to explore more capability in fine tuning.

Please join us on Discord! https://discord.gg/uheqxU6sXN

If you find this tool useful, please consider donating to the project on [Patreon](https://www.patreon.com/everydream).  It is a lot of work to maintain and develop.  Thank you!

## Onward to Every Dream
This trainer is focused on enabling fine tuning with new training data plus weaving in original, ground truth images scraped from the web via Laion dataset or other publically available ML image sets.  Compared to DreamBooth, concepts such as regularization have been removed in favor of adding back ground truth data (ex. Laion), and token/class concepts are removed and replaced by per-image captioning for training, more or less equal to how Stable Diffusion was trained itself. This is a shift back to the original training code and methodology for fine tuning for general cases.

To get the most out of this trainer, you will need to curate a data set to be trained in addition to collect ground truth images to help preserve the model integrity and character.  Luckily, there are additional tools below to help enable that, and will grow over time.

## Techniques

This is a general purpose fine tuning app.  You can train large or small scale with it and everything in between.

Check out [MICROMODELS.MD](./doc/MICROMODELS.MD) for a quickstart guide and example for quick model creation with a small data set.  It is suited for training one or two subects with 20-50 images with no preservation in 10-25 minutes.

Or [README-FF7R.MD](./doc/README-FF7R.MD) for large scale training of many characters with model preservation.

**The trainer now is insensitive to size and aspect ratio of training images with the Multi-Aspect feature!**  Collect your images, use the [Tools](#ground-truth-data-sources-and-data-engineering) to automatically caption your images and go! 

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

You can place all your images in some sort of root training folder and the traniner will recurvisely local and find them all.

You may wish to organize with subfolders so you can adjust your training data mix, something like this:

    /training_samples/MyProject
    /training_samples/MyProject/man
    /training_samples/MyProject/man_laion
    /training_samples/MyProject/man_nvflickr
    /training_samples/MyProject/paintings_laion
    /training_samples/MyProject/drawings_laion

In the above example, "/training_samples/MyProject" will be your root folder for the command line.  

As you build your data set, you may find it is easiest to organize in this way to track your balance between new training data and ground truth used to preserve the model integrity.  For instance, if you have 500 new training images in ../man you may with to use 500  in the /man_laion and another 500 in /man_nvflickr.  You can then experiment by removing different folders to see the effects on training quality and model preservation.  Adding more original ground truth data add  training time, but keep your model from "veering off course" and losing its character. 

### Suggestions

The more data you add from ground truth data sets such as Laion, the more training you will get away with without "damaging" the original model.  The wider variety of data in the ground truth portion of your dataset, the less likely your training images are to "bleed" into the rest of your model, losing qualities like the ability to generate images of other styles you are not training.  This is about knowledge retention in the model by refeeding it the same data it was originally trained on.

## Ground truth data sources and data engineering

Visit [EveryDream Data Engineering Tools](https://github.com/victorchall/EveryDream) to find a **web scraper** that can pull down images from the Laion dataset along with an **Auto Caption** script to prepare your data.  You should consider that your first step before using this trainer.  If you already have data, you can use that, too, but I encourage you to caption your data with that tool for improved training results. 

## Starting training

An example comand to start training:

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t  --actual_resume sd_v1-5_vae.ckpt -n MyProjectName --gpus 0, --data_root training_samples\MyProject

In the above, the source training data is expected to be laid out in subfolders of training_samples\MyProject as described in above sections. It will use the first Nvidia GPU in the system, and resume from the checkpoint named "sd_v1-5_vae.ckpt".  "-n MyProjectName" is merely a name for the folder where logs will be written during training, which appear under /logs. 
## Testing

I strongly recommend attempting to undertrain via the repeats and set max_epoch higher compared to typical dream booth recommendations so you will get a few different ckpts along the course of your training session.  The ckpt files will be dumped to a folder such as _\logs\MyPrject2022-10-25T20-37-40_MyProject_ date stamped to the start of training. There are also test images in the _\logs\images\train_ folder that spit out periodically based on another finetune yaml setting:

      callbacks:
        image_logger:
        target: main.ImageLogger
        params:
            batch_frequency: 300

The images will often not all be fully formed, and are randomly selected based on the last few training images, but it's a good idea to watch those images and learn to understand how they look compared to when you go try your new model out in a normal inference app. 

To continue training on a checkpoint, grab the the desried ckpt file \logs\MyPrject2022-10-25T20-37-40_MyProject\checkpoints and move it back to your base folder and just change the --actual_resume pointer to last.ckpt such as the following:

    python main.py --base configs/stable-diffusion/v1-finetune_everydream.yaml -t  --actual_resume last.ckpt -n MyProjectName --gpus 0, --data_root training_samples\MyProject

If you are close, consider lowering repeats!
## Finetune yaml adjustments

Depending on your project, a few settings may be useful to tweak or adjust.  In [Starting Training](#starting_training) I'm using __v1-finetune_everydream.yaml__ here but you can make your own copies if you like with different adjustments and save them for your projects.  It is a good idea to get familar with this file as tweaking can be useful as you train.

I'll highlight the following settings at the end of the file:

    trainer:
      benchmark: True
      max_epochs: 4
      max_steps: 99000

max_epochs will halt training.  I suggest ending on a clean end of an epoch rather than using a steps limit, so defaults are configured as such.  3-5 epochs will give you a few copies to try. 

      train:
        target: ldm.data.every_dream.EveryDreamBatch
        params:
            set: train
            repeats: 20

Above, the repeats defines the number of times each training image is trained on per epoch.  For large scale training with 500+ images per subject you may find just 10-15 repeats with 3-4 epochs.  As you add more and more data you can slowly use lower repeat values.  For very small training sets, try the micro YAML that has higher repeats (50-100).

You are also free to move data in and out of your training_samples/MyProject folder between training sessions.  If you have multiple subjects and your number of images between them is a bit mismatched in number, say, 100 for one and only 60 for another, you can try running one epoch 25 repeats, then remove the character with 100 images and train just the one with the 60 images for another epoch at 5 repeats.  It's best to try to keep the data evenly spread, but sometimes that is diffcult.  You may also find certain characters are harder to train, and need more on their own.  Again, test!  Go generate images between 

    data:
      target: main.DataModuleFromConfig
      params:
        batch_size: 6

Batch size determine how many images are loaded and trained on in parallel. batch_size 6 will work on a 24GB GPU, 1 will only reduce VRAM use to about 19.5GB.  The batch size will divide the number of steps used as well, but one epoch is still "repeats" number of trainings on each image.  Higher batch sizes are desired to give better generalization as the gradient is calculated across the entire batch.  More images in a batch will also decrease training time by keep your GPU utilization higher.

I recommend not worrying about step count so much. Focus on epochs and repeats instead. 

### Additional notes

Thanks go to the CompVis team for the original training code, Xaiver Xiao for the DreamBooth implementation and tweaking of trainer configs to stuff it into a 24GB card, and Kane Wallmann for code take image captions from the filenames.

References:

[Compvis Stable Diffusion](https://github.com/CompVis/stable-diffusion)

[Xaiver Xiao's DreamBooth implementation](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)

[Kane Wallmann's captioning capability](https://github.com/kanewallmann/Dreambooth-Stable-Diffusion)

# Troubleshooting

**Cuda out of memory:**  You should have <600MB used before starting training to use batch size 6.  People have reported issues with Precision X1 running in the background and Microsoft's system tray weather app causing problems.  You can disable hardware acceleration in apps like Discord and VS Code to reduce VRAM use, and close as many Chrome tabs as you can bear. 

