import timm
import torch
import torch.nn as nn


class TimmNet(nn.Module):
    def __init__(self, model_name: str, embedding_size: int, pretrained: bool):
        super(TimmNet, self).__init__()
        self.model_name = model_name
        self.embedding_size = embedding_size
        self.pretrained = pretrained

        self.net = timm.create_model(
            self.model_name,
            num_classes=self.embedding_size,
            pretrained=self.pretrained,
        )

    def forward(self, x):
        return self.net(x)


def get_model(
    model_name: str, 
    embedding_size: int, 
    pretrained: bool
    ):
    model = TimmNet(
        model_name=model_name,
        embedding_size=embedding_size,
        pretrained=pretrained,
    )
    return model