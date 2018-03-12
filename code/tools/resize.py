import argparse
import os
from PIL import Image


def resize_image(image, size):
    """Resize an image to the given size."""
    return image.resize(size, Image.ANTIALIAS)


def resize_folder_images(image_dir, output_dir, size):
    """Resize the images in 'image_dir' and save into 'output_dir'."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = os.listdir(image_dir)
    num_images = len(images)
    for i, image in enumerate(images):
        with open(os.path.join(image_dir, image), 'r+b') as f:
            with Image.open(f) as img:
                img = resize_image(img, size)
                img.save(os.path.join(output_dir, image), img.format)

        if i % 100 == 0:
            print("[%d/%d] Resized the images and saved into '%s'."
                  % (i, num_images, output_dir))


def resize_images(cf):
    splits = ['train', 'val']
    years = ['2014']

    if not os.path.exists(cf.resized_image_dir):
        os.makedirs(cf.resized_image_dir)

    for split in splits:
        for year in years:
            # build path for input and output dataset
            dataset = split + year
            image_dir = os.path.join(cf.image_dir, dataset)
            output_dir = os.path.join(cf.resized_image_dir, dataset)

            image_size = [cf.resized_image_size, cf.resized_image_size]
            resize_folder_images(image_dir, output_dir, image_size)


