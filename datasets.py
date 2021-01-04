from torch.utils.data import Dataset
import cv2
import glob

# image transformations
import albumentations as A

# parse preprocessing args
import preprocessing

"""
Pytorch Dataset configurations for classification and segmentation.

Also includes a class to create preprocessing arguments using the setup
from a config_reader.PytorchConfig object instance.
"""

def load_img(img_name, grayscale=False):
    """
    Simple wrapper that calls cv2.imread and converts to RGB order if image
    is 3-channel. Load in grayscale images as normal.

    Args:
        img_name (str): Path to image.
        grayscale (bool, optional): Whether image should be treated as
        grayscale, i.e. having 1 channel.

    Raises:
        FileNotFoundError: Path to given image file does not exist or is not valid.

    Returns:
        img (numpy.ndarray): NumPy image in RGB ordering if 3 channels.
    """
    if grayscale:
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # convert to RGB

    # why cv2 doesn't throw FileNotFoundErrors on its own confounds me
    if img is None:
        raise FileNotFoundError(f"{img_name} is not a valid image file")

    return img

class SegmentationDataset(Dataset):
    """
    Pytorch dataset for segmentation tasks. If a path to masks are
    provided, images and masks will be transformed together according
    to the augmentations specified.
    """
    def __init__(self, img_path, mask_path=None, transforms=None,
                    img_preprocessing=None, mask_preprocessing=None,
                    verbose=True):
        # glob paths to images
        self.img_list = sorted(glob.glob(img_path))

        if self.__len__() == 0:
            raise ValueError(f"No images found at {img_path}")
        else:
            if verbose:
                print(f"Found {self.__len()} images in {img_path}")

        # add paths to mask if given
        if mask_path:
            self.mask_list = sorted(glob.glob(mask_path))
            if len(self.img_list) != len(self.mask_list):
                raise ValueError("Number of images and masks don't match!\n" +
                                f"Image path: {img_path}\n" + f"Mask path: {mask_path}")
        else:
            self.mask_list = None

        # image augmentations using albumentations
        self.transforms = transforms

        # preprocessing functions for images and masks
        self.img_preprocessing = img_preprocessing
        if mask_preprocessing is not None and mask_path is None:
            raise ValueError("mask_preprocessing function set but no mask_path given." \
                            "Either provide a mask_path or remove mask_preprocessing argument")
        self.mask_preprocessing = mask_preprocessing

    def __len__(self):
        """
        Return the size of the dataset.
        """
        return len(self.img_list)

    def __getitem__(self, idx):
        """
        Retrieve one entry in dataset and apply preprocessing steps and augmentations.

        Args:
            idx (int): Index of batch in dataset.

        Returns:
            img: Image after augmentations (if given).
            mask (optional): Corresponding mask after augmentations (if given).
        """
        # load image from disk
        img_name = self.img_list[idx]
        img = load_img(img_name)

        # preprocess images
        if self.img_preprocessing:
            img = self.run_preprocessing_functions(img, self.img_preprocessing)

        # prepare mask for batch
        if self.mask_list:
            # load mask from disk
            mask_name = self.mask_list[idx]
            mask = load_img(mask_name, grayscale=True)

            # preprocess masks
            if self.mask_preprocessing:
                mask = self.run_preprocessing_functions(mask, self.mask_preprocessing)

            # perform matching augmentations on both image and mask
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            return img, mask

        else:
            # perform augmentations on image only
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented['image']

            return img

    def run_preprocessing_functions(self, img, func):
        """
        Preprocess image with given function(s). If func is a list,
        img is preprocessed with each function in the list sequentially.

        Args:
            img (array): Image or mask to be preprocessed.
            func (function or list of functions): Image preprocessing function(s).

        Returns:
            img_preprocessed (array): Image after preprocessing.
        """
        # run through sequence of preprocessing functions
        if isinstance(func, list):
            img_preprocessed = img.copy()
            for f in func:
                img_preprocessed = f(img_preprocessed)
        else: # single preprocessing function
            img_preprocessed = f(img)

        return img_preprocessed