import argparse
from config_reader import PytorchConfig
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

# user imports
import datasets
import models

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_name', help='Path to .yml config file')
    args = parser.parse_args()

    # read config file into utility class
    cfg = PytorchConfig(args.config_name)

    # figure out whether to do segmentation or classification
    task = cfg.data.get('TASK')
    if task is None or (task != 'segmentation' and task != 'classification'):
        raise KeyError("Config file must have field TASK be 'segmentation' or 'classification'.")

    if task == 'segmentation':
        # make image/mask datasets for training and validation sets
        train_dataset = datasets.SegmentationDataset(img_path=cfg.paths['train']['images'],
                                                        mask_path=cfg.paths['train']['masks'],
                                                        transforms=cfg.transforms,
                                                        img_preprocessing=cfg.preprocessing_funcs.get('image'),
                                                        mask_preprocessing=cfg.preprocessing_funcs.get('mask'))

        val_dataset = datasets.SegmentationDataset(img_path=cfg.paths['val']['images'],
                                                    mask_path=cfg.paths['val']['masks'],
                                                    transforms=None,
                                                    img_preprocessing=cfg.preprocessing_funcs.get('image'),
                                                    mask_preprocessing=cfg.preprocessing_funcs.get('mask'))

        # build segmentation model with config args
        model = models.SegmentationModel(cfg.model_args, cfg.loss_args, cfg.opt_args, cfg.logging_args)

    # create DataLoaders for training and validation sets
    train_loader = DataLoader(train_dataset, **cfg.dataloader_args)
    val_loader = DataLoader(val_dataset, **cfg.dataloader_args)

    # fit model to dataset
    trainer = Trainer(**cfg.trainer_args)
    trainer.fit(model, train_dataloader=train_loader, val_dataloaders=val_loader)