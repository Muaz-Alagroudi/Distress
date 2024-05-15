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


# def plot_image(image, label=None):
#     """Plots a single image with an optional label"""
#     plt.imshow(image)
#     plt.xlabel(label)
#     plt.xticks(None)
#     plt.yticks(None)
#     plt.tight_layout()
#     plt.show()


def is_image(file_path):
    """Checks if a given file path is a valid image"""
    # Get the file extension
    _, extension = os.path.splitext(file_path)
    # List of common image file extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # Check if the file extension matches any image extensions
    return extension.lower() in image_extensions


def augment_dir(dir_path, depth=0, verbose=False):
    """Recursively go through each directory and augments images along the way"""

    # Iterate through the files in the folder or directory
    for entry in os.scandir(dir_path):
        if entry.is_dir():
            if verbose:
                print("\t" * depth + "In directory ", entry.name)
            augment_dir(os.path.join(dir_path, entry.name), depth + 1)
        else:
            if is_image(entry.name):
                path = os.path.join(dir_path, entry.name)
                if verbose:
                    print("\t" * depth + entry.name)
                augment_image(path, entry.name)


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


def load_images_from_dirs(directory_paths, size=64):
    """Load images from a list of directories and returns a np.array of images
    that are resized/gray scaled and a list of labels"""
    images = []
    labels = []
    for i, directory in enumerate(directory_paths):
        for filename in os.listdir(directory):
            img_path = os.path.join(directory, filename)
            if os.path.isfile(img_path):
                # Load the image using OpenCV
                img = Image.open(img_path)
                img = resize_grayscale(img, size)
                if img is not None:
                    # Convert the image to RGB (OpenCV loads images in BGR format)
                    images.append(img)
                    labels.append(i)
    return np.array(images), np.array(labels)


def get_leaf_directory_paths(root_directory_path):
    """returns leaf directory paths for a root directory"""
    leaf_directory_paths = []
    for dirpath, dirnames,_ in os.walk(root_directory_path):
        if not dirnames:
            leaf_directory_paths.append(dirpath)
    return leaf_directory_paths


def predict_sample(model, sample, class_names):
    dimensions = sample.shape
    sample_reshape = sample.reshape(1, dimensions[1], dimensions[1], 1)
    print(f'sample shape {sample.shape}')
    prediction = model.predict(sample)
    index = np.argmax(prediction)
    class_name = class_names[index]
    dimensions = sample.shape
    sample_reshape = sample.reshape(dimensions[1], dimensions[2], 1)
    plt.figure(figsize=(5, 5))
    plt.imshow(sample_reshape)
    plt.xlabel(class_name)
    plt.xticks(None)
    plt.yticks(None)
    plt.tight_layout()
    plt.show()
