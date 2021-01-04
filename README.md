# Pytorch Classification/Segmentation Pipeline

Got super tired of having to write the same boilerplate code over and over again for segmentation and classification tasks, so I wrote this to standardize my training pipelines and automate everything via a config file (see `segmentation_config.yml` for instance).

Currently this pipeline can only train for segmentation tasks, but I plan to expand it for use in image classification. For the future, I will probably also add a GUI because I am getting too lazy to even edit a .yml config file.

## Dependencies
- Pytorch
- Pytorch Lightning (https://pytorch-lightning.readthedocs.io/en/latest/)
- albumentations (https://albumentations.ai/)
- segmentation-models (https://smp.readthedocs.io/en/latest)