import pytorch_lightning as pl
import torch.optim as optimizers
import segmentation_models_pytorch as smp
import losses

"""
Builds models for classification and segmentation tasks
using Pytorch Lightning.

Segmentation tasks are handled by segmentation-models-pytorch.
"""

class SegmentationModel(pl.LightningModule):
    def __init__(self, model_args, loss_args, opt_args, logging_args):
        super().__init__()
        self.model = self.setup_model(**model_args)
        self.loss_fn = self.setup_loss(**loss_args)
        self.loggers = self.setup_loggers(**logging_args)

    def forward(self, x):
        """
        Forward pass for segmentation model.

        Args:
            x (torch.Tensor): Input tensor for prediction. Needs
            to be in NCHW format (not NHWC)

        Returns:
            z (torch.Tensor): Output predictions from model
        """
        z = self.model(z)
        return z

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)

        # TODO: fix up logging funcs
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        # TODO: fix up logging funcs
        loss = self.loss_fn(y_pred, y)
        self.log('val_loss', loss)

    def configure_optimizers(self, opt_args):
        opt = setup_optimizer(**opt_args)
        return opt

    def setup_model(self, base_model='Unet', **model_kwargs):
        """
        Retrieve base model from segmentation_models_pytorch and instantiate
        with given model keyword arguments.

        Args:
            base_model_name (str): Name of base segmentation model architecture.
            See https://smp.readthedocs.io/en/latest/models.html#unet for more details.

            **model_kwargs: Keyword args for SMP model.

        Raises:
            AttributeError: Base model doesn't exist in segmentation-models-pytorch

        Returns:
            model: Instantiated Pytorch model ready for training.
        """
        try:
            seg_model = getattr(smp, base_model)
        except AttributeError:
            raise ValueError(f"No such model {base_model} in segmentation-models-pytorch")
        # call model with given model args
        model = seg_model(**model_kwargs)
        return model

    def setup_loss(self, name="DiceLoss", **loss_kwargs):
        """
        Instantiate a loss function class, either from smp.losses or from losses.py
        for custom classes. setup_loss() will attempt to retrieve the loss first from
        smp.losses and will search in losses.py next.

        Args:
            name (str, optional): Name of loss function, either from smp.losses or
            from losses.py. Defaults to "DiceLoss".

            See https://smp.readthedocs.io/en/latest/losses.html for more details.

            **loss_kwargs: Keyword args for loss class.

        Raises:
            ValueError: Given loss name cannot be found in either smp.losses
            or in losses.py

        Returns:
            Pytorch loss: Loss function accepting (y_pred, y) to compute scalar loss
        """
        try:
            # attempt to get loss from segmentation-models-pytorch
            loss_fn = getattr(smp.losses, name)
        except AttributeError:
            try:
                # retrieve custom loss from losses.py
                loss_fn = getattr(losses, name)
            except AttributeError:
                raise ValueError(f"No loss function {name} found in segmentation-models-pytorch or in losses.py")

        return loss_fn(**loss_kwargs)

    def setup_optimizer(self, name='Adam', **opt_kwargs):
        """
        Define an optimizer for training/evaluation using optimizer name
        and optional keyword args.

        Args:
            name (str, optional): Optimizer name. Defaults to 'Adam'.
            **opt_kwargs: Keyword args relating to optimizer.

        Raises:
            ValueError: Name of optimizer cannot be found in torch.optim.*

        Returns:
            Pytorch optimizer
        """
        try:
            opt = getattr(optimizers, name)
        except AttributeError:
            raise ValueError(f"No such optimizer {name} found")
        return opt(self.parameters(), **opt_kwargs)

    def setup_loggers(self, **kwargs):
        return