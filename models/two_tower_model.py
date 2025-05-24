import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, user_encoder, item_encoder):
        super(TwoTowerModel, self).__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def forward(self, user_seq, category, store, parent_asin, text_embedding):
        user_vec = self.user_encoder(user_seq)
        item_vec = self.item_encoder(category, store, parent_asin, text_embedding)
        scores = (user_vec * item_vec).sum(dim=-1)  # 点积
        return scores