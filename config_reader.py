# modules referenced outside config
import albumentations
from albumentations.pytorch import ToTensorV2
import preprocessing

# system imports
from warnings import warn
import os
import yaml

class PytorchConfig(object):
    def __init__(self, config_name):
        self.data = self.load_config(config_name)

        # set up paths dict for training, validation, and results
        self.paths = self.parse_paths(self.data['PATHS'])

        # keyword arguments for creating model, loss, optimizer, and logging funcs
        self.model_args = self.data['MODEL']
        self.loss_args = self.data['LOSS']
        self.opt_args = self.data['OPTIMIZER']
        self.logging_args = {} # TODO: implement logging arguments

        # preprocessing funcs and data augmentations
        self.preprocessing_funcs = self.eval_prep_funcs(self.data.get('PREPROCESSING'))
        self.transforms = self.gather_augmentations(self.data.get('AUGMENTATIONS'))

        # dataloader and Trainer (pytorch-lightning) args
        self.dataloader_args = self.data['DATALOADER']
        self.trainer_args = self.data['TRAINER']

    @staticmethod
    def load_config(config_name):
        with open(config_name) as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        return cfg

    def parse_paths(self, paths_dict):
        for key, val in paths_dict.items():
            if isinstance(val, dict):
                root_dir = val.get('root_dir')
                base_path = val.get('base_path')

                if root_dir is not None and base_path is not None:
                    # join root directory and basename to create full path
                    path = os.path.join(root_dir, base_path)
                    paths_dict[key] = path
                else:
                    self.parse_paths(val)
        return paths_dict

    @staticmethod
    def eval_prep_funcs(prep_dict):
        """
        Given a mapping of image types (i.e. "image", "mask") and corresponding preprocessing
        function names, returns a mapping of image types and evaluated preprocessing functions.

        Preprocessing function names must be found in preprocessing.py, or ValueError will be raised.

        Args:
            prep_dict (dict<str, str or list of str>): Maps image type to preprocessing function(s).
            Examples of image types are "image" or "mask" and refer to what array to perform
            the preprocessing functions on. Mapping can lead to either a single function
            str name or a list of preprocessing function names.

        Returns:
            preprocessing_funcs (dict<str, func or list of funcs>): Maps image types ("image", "mask")
            to *evaluated* preprocessing functions.
        """
        if prep_dict is None:
            warn("No preprocessing functions specified. If there are,"
                "make sure there is a 'PREPROCESSING' field in the config file.")
            return None

        # dict to hold image type (i.e. "image," "mask") and what prep funcs to apply to image type
        evaluated_prep_funcs = {}

        def retrieve_prep_func(func_name):
            try:
                func = getattr(preprocessing, func_name)
            except AttributeError:
                raise ValueError(f"Preprocessing function {func_name} not found in preprocessing.py")
            return func

        # iterate over image types to apply preprocessing functions
        for image_type, func_name in prep_dict.items():
            if func_name is not None: # filter out fields which are empty
                # list of preprocessing functions
                if isinstance(func_name, list):
                    try:
                        prep_func = [retrieve_prep_func(f) for f in func_name]
                    except AttributeError:
                        raise
                else:
                    # one single preprocessing function
                    prep_func = retrieve_prep_func(func_name)

                # add key-value pair to dictionary of preprocessing functions
                evaluated_prep_funcs[image_type] = prep_func

        return evaluated_prep_funcs

    @staticmethod
    def gather_augmentations(aug_dict):
        if aug_dict is None:
            warn("Training with no image augmentations")
            return None
        else:
            augs = []
            for transform_name, transform_params in aug_dict.items():
                try:
                    # attempt to get augmentation from albumentations
                    transform_fn = getattr(albumentations, transform_name)

                    # create transformation with given parameters
                    transform = transform_fn(**transform_params)

                    # add to running list of data augmentations
                    augs.append(transform)
                except AttributeError:
                    raise ValueError(f"No data augmentation {transform_name} found in albumentations")

            # convert numpy to Pytorch tensor and combine all augmentations
            augs.append(ToTensorV2())
            augs = albumentations.Compose(augs)
            return augs