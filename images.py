from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os


def resize_grayscale(image, width, height=None):
    """Resizes the image to a specified width and height,
     and returns a grayscale image as a np.array"""
    if height is None:
        height = width
    image = image.resize((width, height))
    image = np.array(image.convert('L'))
    return image


def plot_image(image, label=None):
    """Plots a single image with an optional label"""
    plt.imshow(image)
    plt.xlabel(label)
    plt.xticks(None)
    plt.yticks(None)
    plt.tight_layout()
    plt.show()


def is_image(file_path):
    """Checks if a given file path is a valid image"""
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # Check if the file extension matches any image extensions
    return extension.lower() in image_extensions


def augment_dir(dir_path, depth=0):
    """Recursively go through each directory and augments images along the way"""

    # Iterate through the files in the folder or directory
    for entry in os.scandir(dir_path):
        if entry.is_dir():
            # print("\t" * depth + "In directory ", entry.name)
            augment_dir(os.path.join(dir_path, entry.name), depth + 1)
        else:
            if is_image(entry.name):
                path = os.path.join(dir_path, entry.name)
                # print("\t" * depth + entry.name + " in directory ", dir_path)
                augment_image(path, entry.name)
                # print("\t" * depth + f"an image {entry.name}")


def augment_image(image_path, name=''):
    """ augments a given image by rotating the image 90 and 180 degrees, flipping the
    image horizontally and adjusting the brightness of the image"""
    with Image.open(image_path) as img:
        # rotate, flip, and change the brightness of the image
        rotated = img.rotate(90)
        rotated2 = img.rotate(180)
        flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
        brightness_factor = 1.5
        brightness_img = img.point(lambda p: p * brightness_factor)

        # get parent directory and save the modified image
        parent = os.path.dirname(image_path)

        rotated.save(os.path.join(parent, f'r_{name}.png'))
        rotated2.save(os.path.join(parent, f'r2_{name}.png'))
        flipped_img.save(os.path.join(parent, f'f_{name}.png'))
        brightness_img.save(os.path.join(parent, f'b_{name}.png'))


directory_path = 'Flexible Pavement Distresses'
# augment_dir(directory_path)
augment_dir(directory_path)
