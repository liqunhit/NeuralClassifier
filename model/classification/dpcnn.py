#!usr/bin/env python
# coding:utf8

import torch
import torch.nn.functional as F
 
from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier


class DPCNN(Classifier):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
    """

    def __init__(self, dataset, config):
        super(DPCNN, self).__init__(dataset, config)
        self.num_kernels = config.DPCNN.num_kernels
        self.pooling_stride = config.DPCNN.pooling_stride
        self.kernel_size = config.DPCNN.kernel_size
        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "DPCNN kernel should be odd!"
        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                config.embedding.dimension, self.num_kernels,
                self.kernel_size, padding=self.radius)
        )

        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius)
        ) for _ in range(config.DPCNN.blocks + 1)])

        self.linear = torch.nn.Linear(self.num_kernels, len(dataset.label_map))

    def get_parameter_optimizer_dict(self):
        params = super(DPCNN, self).get_parameter_optimizer_dict()
        params.append({'params': self.convert_conv.parameters()})
        params.append({'params': self.convs.parameters()})
        params.append({'params': self.linear.parameters()})
        return params

    def forward(self, batch):
        if self.config.feature.feature_names[0] == "token":
            embedding = self.token_embedding(
                batch[cDataset.DOC_TOKEN].to(self.config.device))
        else:
            embedding = self.char_embedding(
                batch[cDataset.DOC_CHAR]).to(self.config.device)
        embedding = embedding.permute(0, 2, 1)
        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features
        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features
        doc_embedding = F.max_pool1d(
            conv_features, conv_features.size(2)).squeeze()
        return self.dropout(self.linear(doc_embedding))
