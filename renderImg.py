#!/usr/bin/env python

from keras.preprocessing.image import ImageDataGenerator
from keras.utils.image_utils import img_to_array
from keras.utils import load_img
import numpy as np
from imutils import paths
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="Input directory")
ap.add_argument("-o", "--output", required=True, help="Output directory")
ap.add_argument("-p", "--prefix", type=str, default="image", help="Extensions name")
args = vars(ap.parse_args())


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


print(f"{bcolors.OKBLUE}[INFO] Generating images...{bcolors.ENDC}")
imagePaths = list(paths.list_images(args["input"]))
for i in range(imagePaths.__len__()):
    image = load_img(imagePaths[i])
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    aug = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    imageGen = aug.flow(
        image,
        batch_size=1,
        save_to_dir=args["output"],
        save_prefix=args["prefix"],
        save_format="jpg",
    )

    total = 0
    for image in imageGen:
        total += 1
        if total == 10:
            break

    percent = (i / imagePaths.__len__()) * 50
    done = "=" * int(percent)
    naht = " " * (50 - int(percent))
    print(
        f"{bcolors.OKGREEN}[PROSSESS][{done}{naht}]{imagePaths[i]}{bcolors.ENDC}",
        end="\r",
    )

print(f"{bcolors.OKGREEN}[INFO] Done...{bcolors.ENDC}")
