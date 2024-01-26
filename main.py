import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

class NWHead(nn.Module):
    def forward(self,
                query_feats,
                support_feats,
                support_labels):
        """
        Computes Nadaraya-Watson prediction.
        Returns (softmaxed) predicted probabilities.
        Args:
            query_feats: (b, embed_dim)
            support_feats: (b, num_support, embed_dim)
            support_labels: (b, num_support, num_classes)
        """
        query_feats = query_feats.unsqueeze(1)

        scores = -torch.cdist(query_feats, support_feats)
        probs = F.softmax(scores, dim=-1)
        return torch.bmm(probs, support_labels).squeeze(1)

# Load the CSV file into a DataFrame
df = pd.read_csv('/dataNAS/people/paschali/datasets/chexpert-public/chexpert-public/train.csv')
print(df.head())