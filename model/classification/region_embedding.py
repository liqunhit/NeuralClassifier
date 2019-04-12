#!usr/bin/env python
# coding:utf8

import torch

from dataset.classification_dataset import ClassificationDataset as cDataset
from model.classification.classifier import Classifier
from model.model_util import InitType
from model.model_util import init_tensor


class RegionEmbedding(Classifier):
    """Implement region embedding classification method
    Reference: "A New Method of Region Embedding for Text Classification"
    """

    def __init__(self, dataset, config):
        super(RegionEmbedding, self).__init__(dataset, config)
        self.region_size = config.embedding.region_size
        self.radius = int(self.region_size / 2)
        self.linear = torch.nn.Linear(config.embedding.dimension,
                                      len(dataset.label_map))
        init_tensor(self.linear.weight, init_type=InitType.XAVIER_UNIFORM)
        init_tensor(self.linear.bias, init_type=InitType.UNIFORM, low=0, high=0)

    def get_parameter_optimizer_dict(self):
        params = super(RegionEmbedding, self).get_parameter_optimizer_dict()
        return params

    def forward(self, batch):
        embedding, _, mask = self.get_embedding(
            batch, [self.radius, self.radius], cDataset.VOCAB_PADDING)
        # mask should have same dim with padded embedding
        mask = torch.nn.functional.pad(mask, (self.radius, self.radius, 0, 0), "constant", 0)
        mask = mask.unsqueeze(2)
        embedding = embedding * mask
        doc_embedding = torch.sum(embedding, 1)
        doc_embedding = self.dropout(doc_embedding)
        return self.linear(doc_embedding)
