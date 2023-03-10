import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig

from .dataset import DefectDataset
from .transformation import Transforms


class DefectDataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_val, y_val, cfg: DictConfig):
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.cfg = cfg

    def setup(self, stage=None):
        self.train_dataset = DefectDataset(
            image_name_list=self.x_train,
            label_list=self.y_train,
            img_dir=self.cfg.img_dir,
            transform=Transforms(cfg=self.cfg),
            phase='train'
        )
        self.val_dataset = DefectDataset(
            image_name_list=self.x_val,
            label_list=self.y_val,
            img_dir=self.cfg.img_dir,
            transform=Transforms(cfg=self.cfg),
            phase='val'
        )

    def train_dataloader(self):   
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train_dataloader.batch_size,
            shuffle=self.cfg.train_dataloader.shuffle,
            num_workers=self.cfg.train_dataloader.num_workers,
            pin_memory=self.cfg.train_dataloader.pin_memory,
            )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.val_dataloader.batch_size,
            shuffle=self.cfg.val_dataloader.shuffle,
            num_workers=self.cfg.val_dataloader.num_workers,
            pin_memory=self.cfg.val_dataloader.pin_memory,
            )
