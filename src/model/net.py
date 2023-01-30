import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig

from .model import get_model
from .loss import get_loss_fn
from .optimizer import get_optimizer


class Net(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        # model
        self.model = get_model(
            model_name=self.cfg.model.model_name,
            embedding_size=self.cfg.embedding_size,
            pretrained=self.cfg.model.pretrained,
        )
        
        # loss function
        self.loss_fn = get_loss_fn(cfg=self.cfg)

        # Important: This property activates manual optimization.
        self.automatic_optimization = False
    

    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        optimizer, loss_optimizer = self.optimizers()
        optimizer.zero_grad()
        loss_optimizer.zero_grad()

        x, t = batch
        y = self(x)
        loss = self.loss_fn(y, t)

        # instead of loss.backward()
        self.manual_backward(loss)

        optimizer.step()
        loss_optimizer.step()

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}
    

    def validation_step(self, batch, batch_idx):
        x, t = batch
        y = self(x)
        loss = self.loss_fn(y, t)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'loss': loss}
    

    def configure_optimizers(self):
        optimizer = get_optimizer(cfg=self.cfg, net=self.model)
        loss_optimizer = get_optimizer(cfg=self.cfg, net=self.loss_fn)
        return [optimizer, loss_optimizer]