from pytorch_metric_learning import losses, distances, regularizers
from omegaconf import DictConfig


def get_loss_fn(cfg: DictConfig):
    if cfg.loss_fn.name == 'ArcFaceLoss':
        distance = distances.CosineSimilarity()
        regularizer = regularizers.RegularFaceRegularizer()
        loss_fn = losses.ArcFaceLoss(
            num_classes=cfg.num_classes,
            embedding_size=cfg.embedding_size,
            margin=28.6,
            scale=64,
            weight_regularizer=regularizer, 
            distance=distance
        )
        return loss_fn
    
    elif cfg.loss_fn.name == 'SubCenterArcFaceLoss':
        loss_fn = losses.SubCenterArcFaceLoss(
            num_classes=cfg.num_classes,
            embedding_size=cfg.embedding_size,
            margin=28.6, 
            scale=64, 
            sub_centers=3,
        )
        return loss_fn
    
    else:
        raise ValueError(f'Unknown optimizer: {cfg.loss_fn.name}')