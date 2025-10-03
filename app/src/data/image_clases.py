"""Image classes.

Represents images and their transformations.
"""
import os
from enum import Enum

from src.data import (
    add_noise_to_image,
    add_random_circles,
    invert_colors,
    mirror_image,
    random_subimage,
    rotate_image,
    set_brightness,
    shift_color_towards_random,
    shuffle_image,
    split_image_into_grid,
)
from src.utils import load_image


class Transformation(Enum):
    """Enum class that defines a set of image transformation operations.

    Each transformation is represented by a name and a corresponding function
    that applies the transformation to an input image.

    Transformations include:
        - SPLIT: Splits the image into smaller grid pieces.
        - ROTATE: Rotates the image by a random degree.
        - MIRROR: Mirrors the image horizontally.
        - SUB: Extracts a random subimage from the original image.
        - SHUFFLE: Shuffles the pixels of the image.
        - CIRCLES: Adds random circles to the image.
        - BRIGHTNESS: Adjusts the brightness of the image.
        - INVERT: Inverts the colors of the image.
        - SHIFT_COLOR: Shifts the color tones of the image randomly.
        - NOISE: Adds random noise to the image.
    """

    SPLIT = ("_spli", lambda img_in, n_pieces: split_image_into_grid(img_in, n_pieces))

    ROTATE = ("_rot", lambda img_in, seed: rotate_image(img_in, seed))
    MIRROR = ("_mir", lambda img_in, seed: mirror_image(img_in))
    SUB = ("_sub", lambda img_in, seed: random_subimage(img_in, seed))

    SHUFFLE = ("_shu", lambda img_in, seed: shuffle_image(img_in, seed))
    CIRCLES = ("_cir", lambda img_in, seed: add_random_circles(img_in, seed))
    BRIGHTNESS = ("_bri", lambda img_in, seed: set_brightness(img_in, seed))

    INVERT = ("_inv", lambda img_in, seed: invert_colors(img_in))
    SHIFT_COLOR = ("_shi", lambda img_in, seed: shift_color_towards_random(img_in, seed))
    NOISE = ("_noi", lambda img_in, seed: add_noise_to_image(img_in, seed))

    def pipe_to_name(pipeline):
        """Convert a list of transformations to a string representation of their combined names.

        Args:
            pipeline (list): A list of transformations to be converted into a name string.

        Returns:
            str: A concatenated string containing the names of the transformations applied.
        """
        name = ""
        if Transformation.ROTATE in pipeline:
            name += Transformation.ROTATE.value[0]
        if Transformation.MIRROR in pipeline:
            name += Transformation.MIRROR.value[0]
        if Transformation.SUB in pipeline:
            name += Transformation.SUB.value[0]
        if Transformation.SHUFFLE in pipeline:
            name += Transformation.SHUFFLE.value[0]
        if Transformation.CIRCLES in pipeline:
            name += Transformation.CIRCLES.value[0]
        if Transformation.BRIGHTNESS in pipeline:
            name += Transformation.BRIGHTNESS.value[0]
        if Transformation.INVERT in pipeline:
            name += Transformation.INVERT.value[0]
        if Transformation.SHIFT_COLOR in pipeline:
            name += Transformation.SHIFT_COLOR.value[0]
        if Transformation.NOISE in pipeline:
            name += Transformation.NOISE.value[0]
        return name

    def name_to_pipe(name: str):
        """Convert string representation of a combined transforms name back into transformas.

        Args:
            name (str):
                A string representing the transformations applied, where each
                transformation is represented by its identifier (e.g. "_rot" -> rotation).

        Returns:
            list: A list of transformations corresponding to the input name string.
        """
        pipe = []
        name_split = name.split("_")
        name_split = ["_" + part for part in name_split if part]
        if Transformation.ROTATE.value[0] in name_split:
            pipe.append(Transformation.ROTATE)
        if Transformation.MIRROR.value[0] in name_split:
            pipe.append(Transformation.MIRROR)
        if Transformation.SUB.value[0] in name_split:
            pipe.append(Transformation.SUB)
        if Transformation.SHUFFLE.value[0] in name_split:
            pipe.append(Transformation.SHUFFL)
        if Transformation.CIRCLES.value[0] in name_split:
            pipe.append(Transformation.CIRCLES)
        if Transformation.BRIGHTNESS.value[0] in name_split:
            pipe.append(Transformation.BRIGHTNESS)
        if Transformation.INVERT.value[0] in name_split:
            pipe.append(Transformation.INVERT)
        if Transformation.SHIFT_COLOR.value[0] in name_split:
            pipe.append(Transformation.SHIFT_COLOR)
        if Transformation.NOISE.value[0] in name_split:
            pipe.append(Transformation.NOISE)
        return pipe


class SampleImage:
    """Images class for sample and groundtruth.

    A class that represents a single sample point consisting of an input
    image and its corresponding ground truth.
    """

    def __init__(
        self,
        path_img_x: str,
        path_img_y: str,
        load_images: bool = False,
        keep_in_memory: bool = False,
    ):
        self.path_img_x = path_img_x
        self.path_img_y = path_img_y
        self.img_x = None
        self.img_y = None

        if load_images:
            self.get_images(keep_in_memory=keep_in_memory)

    def get_paths(self):
        """Retrieve both the input image and the ground truth as a tuple.

        Returns:
            tuple: A tuple containing the PATH input image and the ground truth.
        """
        return self.path_img_x, self.path_img_y

    def get_names(self):
        """Retrieve both the input image and the ground truth file names as a tuple.

        Returns:
            tuple: A tuple containing the file names of the input image and the ground truth.
        """
        return os.path.basename(self.path_img_x), os.path.basename(self.path_img_y)

    def get_images(self, keep_in_memory: bool = True):
        """Retrieve both the input image and the ground truth as a tuple.

        Returns:
            tuple: A tuple containing the input image and the ground truth.
        """
        if self.img_x is not None and self.img_y is not None:
            return self.img_x, self.img_y

        img_x = load_image(self.path_img_x)
        img_y = load_image(self.path_img_y, to_bw=True)

        if keep_in_memory:
            self.img_x = img_x
            self.img_y = img_y

        return img_x, img_y

    def save_images(self, img_dir: str, gt_dir: str) -> None:
        """Save both the input image and the ground truth to the specified directory.

        Args:
            dir (str): The directory where the images will be saved.
        """
        if self.img_x is None or self.img_y is None:
            self.get_images(keep_in_memory=True)

        base_name_x, base_name_y = self.get_names()

        # Save images
        self.img_x.save(os.path.join(img_dir, base_name_x))
        self.img_y.save(os.path.join(gt_dir, base_name_y))

        # Free memory
        self.img_x.close()
        self.img_y.close()
        self.img_x = None
        self.img_y = None
