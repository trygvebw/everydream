# Every Dream trainer for Stable Diffusion

This is a bit of a divergence from other fine tuning methods out there for Stable Diffusion.  No more "DreamBooth" stuff like tokens, classes, or regularization, though I thank the DreamBooth training community for sharing information and techniques.  Yet, it is time to move on.

## Onward to Every Dream
This trainer is focused on enabling fine tuning with new training data plus weaving in original, ground truth images scraped from the web via Laion dataset or other publically available ML image sets.  Compared to DreamBooth, concepts such as regularization have been removed, an token/class are no long concepts used, as they have been replaced by per-image captioning for training, more or less equal to how Stable Diffusion was trained itself. This is a shift back to the original training code and methodology for fine tuning for general cases.

To get the most out of this trainer, you will need to curate a data set to be trained in addition to ground truth images to help preserve the model integrity and character.  Luckily, there are additional tools below to help enable that, and will grow over time.

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

In the above example, "/training_samples/MyProject" will be your root folder for the command line.  It must be devoid of anything but the subfolders.  **The subfolders again are purely for your own organizational purposes, the names of the subfolders do not matter to the trainer.** It's up to you how you want to name or organize subfolders, the only requirement is that you use a single layer of subfolders and the root folder for your project contains nothing but the subfolders.  You must not put images directly into /training_samples/MyProject. 

Also in the above example, /training_samples/MyProject/man would contain new training images you want to "teach" the model, and the man_laion and man_nvflickr sets would contain images scraped from laion or other original sources (see below for possible sources). It's up to you what you want to include.

As you build your data set, you may find it is easiest to organize in this way to track your balance between new training data and ground truth used to preserve the model integrity.  For instance, if you have 500 new training images in ../man you may with to use 500  in the /man_laion and another 500 in /man_nvflickr.  You can then experiment by removing different folders to see the effects on training quality.

### Suggestions

The more data you add from ground truth data sets such as Laion, the more training you will get away with without "damaging" the original model.  The wider variety of data in the ground truth portion of your dataset, the less likely your training images are to "bleed" into the rest of your model, losing qualities like the ability to generate images of other styles you are not training.  This is about knowledge retention in the model by refeeding it the same data it was originally trained on.

## Ground truth data sources and data engineering

Visit [EveryDream Data Engineering Tools](https://github.com/victorchall/EveryDream) to find a web scraper that can pull down images from the Laion dataset.  You should consider that your first step before using this trainer to collect data.

I suggest pulling down all the files for this set in particular: [https://huggingface.co/datasets/laion/laion2B-en-aesthetic](https://huggingface.co/datasets/laion/laion2B-en-aesthetic) to use with the web scraper.

There is information is in the EveryDream Data Engineering Tools link above on how to run the web scrape.  The webscrape takes zero GPU power, so you can run it locally on any PC with Python before renting GPU power if needed.

The Nvidia Flickr set is also helpful and in a fairly "ready to use" format besides renaming the files: [https://github.com/NVlabs/ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) 
For this trainer, I suggest "close up photo of a person" for captioning of this dataset. If you want, you can go further and separate male/female photos and caption them "close up phot of a man" or "..a woman" as you see fit.  "a close up of a person" is also acceptable, dropping "photo".  You can simply select all in windows, F2 to rename and type "a close up of a person_" **without the quotes but with the underscore** to format the filename captions in a way this trainer can use.

Thanks to Xaiver Xiao for the DreamBooth implementation and tweaking of trainer configs to stuff it into a 24GB card, and Kane Wallmann for code take image captions from the filenames.

### Additional notes

References:

[Xaiver Xiao's DreamBooth implementation](https://github.com/XavierXiao/Dreambooth-Stable-Diffusion)

[Kane Wallmann's captioning capability](https://github.com/kanewallmann/Dreambooth-Stable-Diffusion)