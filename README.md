## Semi-Automated Annotation

This is expermental project to assit annotation process with **image-guided search** from OWL-VIT model.

- [OWL-VIT](https://huggingface.co/docs/transformers/en/model_doc/owlvit)
- [Paper](https://arxiv.org/abs/2205.06230)

## Data set up

1. Crop objects from image ( ~ 10 images) and put under the object_patches folder
with class name folder 
  Folder structure 
    - object_patches/
        - class_name/
            - object_img1.jpg
            - object_img2.jpg
            .
            .

2. Create raw_dataset folder under datasets folder and put raw/unlabeled images 
    - datasets/
        - raw_dataset/
            - raw_img1.jpg
            - raw_img2.jpg 
            .
            .


## Runing the system

Edit folder paths in annotate.py ( main function )

```
python annotate.py
```
