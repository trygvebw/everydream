This is a finetuning of the RunwayML 1.5 ckpt. 

Info: https://huggingface.co/panopstor/ff7r-stable-diffusion

Model: https://huggingface.co/panopstor/ff7r-stable-diffusion/blob/main/ff7r-v5-1.ckpt


This fine tuning includes over a dozen concepts trained in 1636 images with individual captions on each image, plus 1636 images scraped from the Laion dataset for model preservation. 

See the Hugging Face link for more info and a downloadable model you can try.

Everything should be fairly natural to prompt with the exception of Biggs and Wedge which were trained with "ff7r" as it if were their surname.

Prompt guide:

    cloud strife
    tifa lockhart
    barret wallace
    aerith gainsborough
    biggs ff7r
    wedge ff7r
    in midgar city
    slums district of midgar city
    business district of midgar city
    train station in midgar city (optionally in a district name from above)
    streets of midgar city (optional in a district name from above)
    shinra security officer holding an assault rifle
    rooftops of the midgar slums district
    food truck in the slums district of midgar

You can mix and match all the above!  Try it out! 

Preservation tests.

1.4 is far left, 1.5 far right.

FF7R 4.1 was trained on Kane Wallmann's repo with 1400 images training and 1400 images of regulation generated out of SD itself, similar but not quite equal to dreambooth technique.

FF7R 5.1 was trained on EveryDream with 1636 images training and 1636 images of Laion ground truth without DreamBooth techniques.

(warning huge file!)

![a](./demo/mega_test01.webp)

New characters tests:
![a](./demo/mega_test_characters01.webp)
