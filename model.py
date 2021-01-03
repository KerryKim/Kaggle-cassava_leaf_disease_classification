import timm

import torch
import torch.nn as nn

##
class CassvaImgClassifier(nn.Module):
    def __init__(self, network, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(network, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

        '''
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            #nn.Linear(n_features, hidden_size,bias=True), nn.ELU(),
            nn.Linear(n_features, n_class, bias=True)
        )
        '''

    def forward(self, x):
        x = self.model(x)
        return x