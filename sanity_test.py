import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import callbacks
import hydra
from omegaconf import DictConfig

from src.dataset import DefectDataModule
from src.model import Net


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):

    # read data frame
    train_master = pd.read_csv(cfg.train_master_dir)

    train_num_classes = train_master['target'].nunique()
    assert (
        cfg.num_classes == train_num_classes
    ), f'num_classes should be {train_num_classes}'

    # image name list
    image_name_list = train_master['id'].values

    # label list
    label_list = train_master['target'].values

    # split train & val
    x_train, x_val, y_train, y_val = train_test_split(
        image_name_list,
        label_list,
        test_size=cfg.train_test_split.test_size,
        stratify=label_list,
        random_state=cfg.train_test_split.random_state
    )

    # set seed
    pl.seed_everything(seed=cfg.seed)

    # directory to save models
    SAVE_MODEL_PATH = cfg.save_model_dir
    os.makedirs(SAVE_MODEL_PATH, exist_ok=True)

    # datamodule
    datamodule = DefectDataModule(
        x_train=x_train,
        y_train=y_train,
        x_val=x_val,
        y_val=y_val,
        cfg=cfg
    )

    # net
    net = Net(cfg=cfg)

    # Callbacks
    callback_list = []
    if cfg.callbacks.early_stopping.enable:
        earlystopping = callbacks.EarlyStopping(
            monitor=cfg.callbacks.early_stopping.monitor,
            patience=cfg.callbacks.early_stopping.patience,
            mode=cfg.callbacks.early_stopping.mode,
            verbose=True,
            strict=True,
        )
        callback_list.append(earlystopping)
    if cfg.callbacks.model_checkpoint.enable:
        model_checkpoint = callbacks.ModelCheckpoint(
            dirpath=SAVE_MODEL_PATH,
            filename=f'{cfg.model.model_name}',
            monitor=cfg.callbacks.model_checkpoint.monitor,
            mode=cfg.callbacks.model_checkpoint.mode,
            save_top_k=cfg.callbacks.model_checkpoint.save_top_k,
            save_last=cfg.callbacks.model_checkpoint.save_last,
        )
        callback_list.append(model_checkpoint)
    
    # trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        callbacks=callback_list,
        gpus=cfg.trainer.gpus,
        auto_lr_find=cfg.trainer.auto_lr_find,
        deterministic=cfg.trainer.deterministic
    )

    # train
    trainer.fit(net, datamodule=datamodule)


if __name__ == "__main__":
    main()