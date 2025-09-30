"""Image transformation.

Functions and classes for image transformations and sample representation.
"""
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps


# DIVISION N x X
def split_image_into_grid(image: Image.Image, n: int) -> list[Image.Image]:
    """Split image into n x n grid.

    Splits an image into an n x n grid, scales each subimage to 400x400 pixels,
    and returns them as a list of PIL Image objects.

    Args:
        image (Image.Image): The input image to be spl This should be a PIL Image object.
        n (int): The number of rows and columns to split the image into: n x n grid of subimages.

    Returns:
        List[Image.Image]
            A list of PIL Image objects representing the subimages. Subimages are scaled to 400x400.

    Notes:
        If the dimensions of the input image are not perfectly divisible by n,
        the subimages will still be of the computed size, but this might lead to some
        cropping. The final size of each subimage is fixed at 400x400 pixels.

    Exmaple
        from src.data import image_transformation as it
        num = "001"
        image_in = f"./data/images/satImage_0000{num}.png"
        truth_in = f"./data/groundtruth/satImage_0000{num}.png"
        out_dir_image = f"./data/exp_image"
        out_dir_truth = f"./data/exp_truth"

        split_image_into_subimages(image_in, out_dir_image, 0)
        split_image_into_subimages(truth_in, out_dir_truth, 0)
    """
    # Open the image
    img = image.copy()
    width, height = img.size

    # Calculate the dimensions of each subimage
    sub_width, sub_height = width // n, height // n

    # Generate the subimages
    subimages = []
    for row in range(n):
        for col in range(n):
            # Define the bounding box for the current subimage
            left = col * sub_width
            upper = row * sub_height
            right = left + sub_width
            lower = upper + sub_height
            box = (left, upper, right, lower)

            # Crop and scale the subimage
            sub_img = img.crop(box)
            sub_img = sub_img.resize((400, 400))
            subimages.append(sub_img)

    return subimages


# ROTATION
def rotate_image(image, seed):
    """Rotate image and crop to original size.

    Rotates an image by a specified number of degrees, crops it to the original size,
    and fills blank space with black.

    Args:
        image (Image.Image): The input image to be rotated.
        seed (int): The random seed used for generating the rotation angle.

    Returns:
        Image.Image:
            A PIL Image object representing the rotated and cropped image.

    Example:
        from src.data import image_transformation as it
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        degrees = 45                           # Rotation angle in degrees (counter-clockwise)
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        rotate_image_crop(img_in, degrees, out_dir_img, output_name="img_45_rot.png")
        rotate_image_crop(truth_in, degrees, out_dir_truth, output_name="truth_45_rot.png")
    """
    # Open the image
    img = image.copy()
    original_size = img.size  # Original dimensions (width, height)

    # Create a larger canvas with a black background
    larger_canvas = Image.new("RGB", img.size, (0, 0, 0))  # Black background
    larger_canvas.paste(img, (0, 0))

    # Rotate the image
    random.seed(seed)  # Set the seed for reproducibility
    degrees = random.randint(1, 360)  # nosec B311
    rotated_img = larger_canvas.rotate(degrees, resample=Image.BICUBIC, expand=False)

    # Crop back to the original size
    rotated_cropped_img = rotated_img.crop((0, 0, original_size[0], original_size[1]))

    return rotated_cropped_img


# MIRROR (HORIZONTAL FLIP)
def mirror_image(image):
    """Mirror the content of an image (flips it horizontally).

    Args:
        image (Image.Image): The input image to be mirrored. This should be a PIL Image object.

    Returns:
        Image.Image: The mirrored image as a PIL Image object, flipped horizontally.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        mirror_image(img_in, out_dir_img, output_name="mirrored_image.png")
        mirror_image(truth_in, out_dir_truth, output_name="mirrored_image.png")
    """
    # Open the image
    img = image.copy()

    # Flip the image horizontally (mirror effect)
    mirrored_img = ImageOps.mirror(img)

    return mirrored_img


# STRETCH
def random_subimage(image, seed):
    """Get subimage of random size and position and scale to 400x400.

    Selects a random subimage from the input image using the given seed.
    The subimage size is determined randomly between 10 and 400 using the seed.
    The subimage is then scaled to 400x400 pixels.

    Args:
        image (Image.Image): The input image from which a random subimage will be selected.
        seed (int): The seed value for random number generation, ensuring reproducibility.

    Returns:
        Image.Image: The randomly selected and scaled subimage as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        random_subimage(img_in, 12345, out_dir_img, output_name="subimage.png")
        random_subimage(truth_in, 12345, out_dir_truth, output_name="subimage.png")
    """
    # Set the random seed for reproducibility
    random.seed(seed)

    # Get image dimensions
    width, height = image.size

    # Generate random subimage dimensions between 10 and 400
    subimage_size = random.randint(10, 400)  # nosec B311

    # Calculate the max allowable top-left corner coordinates
    max_x = width - subimage_size
    max_y = height - subimage_size

    # Randomly select the top-left corner of the subimage
    x = random.randint(0, max_x)  # nosec B311
    y = random.randint(0, max_y)  # nosec B311

    # Define the bounding box for the subimage
    box = (x, y, x + subimage_size, y + subimage_size)

    # Crop the subimage from the original image
    subimage = image.crop(box)

    # Resize the subimage to 400x400 pixels
    subimage = subimage.resize((400, 400), Image.Resampling.LANCZOS)

    return subimage


# PUZZLE SHUFFLE
def shuffle_image(image, seed):
    """Divide image into n x n grid and shuffle.

    Divides an image into an n x n grid, shuffles the pieces randomly with a given seed,
    and reconstructs the shuffled image.

    Args:
        image (Image.Image): The input image to be shuffled.
        seed (int): The seed for random shuffling. This ensures reproducibility if provided.

    Returns:
        Image.Image: The shuffled image as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        n = 4                           # Number of divisions (4x4 grid)
        seed = 42                       # Seed for reproducibility
        out_dir_img = "data/exp_image"   # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        shuffle_image(img_in, seed)
        shuffle_image(truth_in, seed)
    """
    # Open the image
    img = image.copy()
    width, height = img.size

    # Set the seed for reproducibility
    random.seed(seed)
    n = random.randint(2, 5)  # nosec B311

    # Calculate the size of each grid cell
    grid_width = width // n
    grid_height = height // n

    # Extract grid cells
    grid_cells = []
    for i in range(n):
        for j in range(n):
            left = j * grid_width
            upper = i * grid_height
            right = left + grid_width
            lower = upper + grid_height
            grid_cells.append(img.crop((left, upper, right, lower)))

    # Set the random seed for reproducibility
    if seed is not None:
        random.seed(seed)

    # Shuffle the grid cells
    random.shuffle(grid_cells)

    # Create a new blank image to reconstruct the shuffled image
    shuffled_img = Image.new("RGB", (width, height))

    # Place the shuffled cells back into the new image
    for i in range(n):
        for j in range(n):
            left = j * grid_width
            upper = i * grid_height
            shuffled_img.paste(grid_cells.pop(0), (left, upper))

    return shuffled_img


# HOLES
def add_random_circles(image, seed):
    """Add black circles to an image at random positions with random diameters.

    Args:
        image (Image.Image): The input image to which circles will be added.
        seed (int): The seed for random diameter and position generation. Ensures reproducibility.

    Returns:
        Image.Image: The modified image with added black circles as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        n = 4                           # Rotation angle in degrees (counter-clockwise)
        seed = 42                       # Seed for reproducibility
        out_dir_img = "data/exp_image"   # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        add_random_circles(img_in, 16, seed, out_dir_img, output_name="img_with_circles.png")
        add_random_circles(truth_in, 16, seed, out_dir_truth, output_name="img_with_circles.png")
    """
    # Open the image
    img = image.copy()
    width, height = img.size

    # Maximum circle diameter
    max_diameter = min(width, height) // 8

    # Set the random seed for reproducibility
    random.seed(seed)
    num_circles = random.randint(1, 32)  # nosec B311

    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Generate and draw circles
    for _ in range(num_circles):
        # Random diameter
        diameter = random.randint(1, max_diameter)  # nosec B311

        # Random position ensuring the circle stays within bounds
        x = random.randint(0, width - diameter)  # nosec B311
        y = random.randint(0, height - diameter)  # nosec B311

        # Draw the circle
        draw.ellipse([x, y, x + diameter, y + diameter], fill="black")

    return img


# BRIGHTNESS
def set_brightness(image, seed):
    """Adjust the brightness of an image based on the input brightness level.

    Args:
        image (Image.Image): The input image to adjust the brightness of.
        seed (int): The seed for random brightness level generation. Ensures reproducibility.

    Returns:
        Image.Image: The adjusted image with modified brightness as a PIL Image object.
        The adjusted image with modified brightness as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        seed = 80                      # Random seed for brightness adjustment
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        set_brightness(img_in, seed, out_dir_img, output_name="brightness_adjusted.png")
        set_brightness(truth_in, seed, out_dir_truth, output_name="brightness_adjusted.png")
    """
    # Brightness
    random.seed(seed)
    brightness_level = random.randint(0, 100)  # nosec B311

    # Open the image
    img = image.copy()

    # Calculate brightness factor
    # 0 -> 0.0 (black), 50 -> 1.0 (no change), 100 -> 2.0 (white)
    brightness_factor = brightness_level / 50

    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    adjusted_img = enhancer.enhance(brightness_factor)

    return adjusted_img


# INVERT
def invert_colors(image):
    """
    Inverts the colors of an image.

    Args:
        image (Image.Image): The input image to invert the colors of.

    Returns:
        Image.Image: The image with inverted colors as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        invert_colors(img_in, out_dir_img, output_name="inverted_image.png")
        invert_colors(truth_in, out_dir_truth, output_name="inverted_image.png")
    """
    # Open the image
    img = image.copy()

    # Invert colors
    inverted_img = ImageOps.invert(img.convert("RGB"))

    return inverted_img


# COLORS
def shift_color_towards_random(image, seed):
    """
    Shifts the color scale of the image towards a randomly generated color based on the seed.

    Args:
        image (Image.Image): The input image to shift the colors of.
        seed (int): The seed used to generate the random color for the color shift.

    Returns:
        Image.Image: The color-shifted image as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        seed = 32
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        shift_color_towards_random(img_in, seed, out_dir_img, output_name="color_shifted.png")
        #shift_color_towards_random(truth_in, seed, out_dir_truth, output_name="color_shifted.png")
    """
    # Open the image
    img = image.copy()

    # Set the random seed for reproducibility
    random.seed(seed)

    # Generate a random color (R, G, B) where each channel is between 0 and 255
    random_color = (
        random.randint(0, 255),  # Red channel # nosec B311
        random.randint(0, 255),  # Green channel # nosec B311
        random.randint(0, 255),  # Blue channel # nosec B311
    )

    # Split the image into RGB channels
    r, g, b = img.split()

    # Calculate the scaling factors to shift the color balance
    r_scale = random_color[0] / 255.0
    g_scale = random_color[1] / 255.0
    b_scale = random_color[2] / 255.0

    # Apply the scaling factors to each color channel
    r = r.point(lambda i: i * r_scale)
    g = g.point(lambda i: i * g_scale)
    b = b.point(lambda i: i * b_scale)

    # Merge the modified channels back into an image
    shifted_img = Image.merge("RGB", (r, g, b))

    return shifted_img


# NOISE
def add_noise_to_image(image, seed):
    """Add random noise to an image.

    Args:
        image (Image.Image): The input image to which noise will be added.
        seed (int): The seed used to generate random noise.

    Returns:
        Image.Image: The noisy image as a PIL Image object.

    Examples:
        img_in = "data/images/satImage_0000001.png"  # Replace with your image file
        truth_in = "data/groundtruth/satImage_0000001.png"  # Replace with your image file
        noise = 100
        out_dir_img = "data/exp_image"  # Replace with your desired output folder
        out_dir_truth = "data/exp_truth"  # Replace with your desired output folder

        # Call the function
        add_noise_to_image(img_in, noise, out_dir_img, output_name="noisy_image.png")
    """
    # Open the image and convert it to a NumPy array
    img = image.copy()
    img_array = np.array(img)

    random.seed(seed)
    noise_level = random.randint(0, 255)  # nosec B311

    # Generate random noise
    noise = np.random.randint(
        -noise_level, noise_level + 1, img_array.shape, dtype=np.int16
    )  # nosec B311

    # Add noise to the image and clip the values to keep them valid (0-255)
    noisy_img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    # Convert the noisy image array back to a PIL Image
    noisy_img = Image.fromarray(noisy_img_array)

    return noisy_img
