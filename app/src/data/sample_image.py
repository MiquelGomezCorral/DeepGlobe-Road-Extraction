"""Image classes.

Represents images and their transformations.
"""
import os

from PIL import Image
from src.utils import load_image


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

    def get_paths(self) -> tuple[str, str]:
        """Retrieve both the input image and the ground truth as a tuple.

        Returns:
            tuple[str, str]: A tuple containing the PATH input image and the ground truth.
        """
        return self.path_img_x, self.path_img_y

    def get_names(self) -> tuple[str, str]:
        """Retrieve both the input image and the ground truth file names as a tuple.

        Returns:
            tuple[str, str]. File names of the input image and the ground truth.
        """
        return os.path.basename(self.path_img_x), os.path.basename(self.path_img_y)

    def get_images(self, keep_in_memory: bool = False) -> tuple[Image.Image, Image.Image]:
        """Retrieve both the input image and the ground truth as a tuple.

        Returns:
            tuple[Image.Image, Image.Image]: input image and the ground truth.
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
            self.get_images(keep_in_memory=False)

        base_name_x, base_name_y = self.get_names()

        # Save images
        self.img_x.save(os.path.join(img_dir, base_name_x))
        self.img_y.save(os.path.join(gt_dir, base_name_y))

        # Free memory
        self.img_x.close()
        self.img_y.close()
        self.img_x = self.img_y = None

    def set_images(self, img_x: Image.Image, img_y: Image.Image) -> None:
        """Set the input image and ground truth.

        Args:
            img_x (Image.Image): The input image.
            img_y (Image.Image): The ground truth image.
        """
        self.img_x = img_x
        self.img_y = img_y
