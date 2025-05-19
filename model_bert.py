import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import random
import time

# ========= NeuMF =========
# --------- change embedding ways -----------
class NeuMF(nn.Module):
    def __init__(self, num_users, num_items, item_attr_dim=768, id_emb_dim=64, mlp_layers=[128, 64]):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, id_emb_dim)
        self.item_id_embedding = nn.Embedding(num_items, id_emb_dim)  # 原始 ID embedding

        self.item_attr_dim = item_attr_dim
        self.id_emb_dim = id_emb_dim

        # 拼接 ID embedding + BERT embedding
        input_dim = id_emb_dim + item_attr_dim
        mlp = []
        for h in mlp_layers:
            mlp.append(nn.Linear(input_dim, h))
            mlp.append(nn.ReLU())
            input_dim = h
        self.mlp_layers = nn.Sequential(*mlp)

        self.output = nn.Linear(input_dim, 1)

    def forward(self, user_ids, item_ids, item_attrs):  # item_attrs: [batch_size, 768]
        user_emb = self.user_embedding(user_ids)         # [batch_size, D_id]
        item_id_emb = self.item_id_embedding(item_ids)   # [batch_size, D_id]
        item_emb = torch.cat([item_id_emb, item_attrs], dim=-1)  # 拼接 → [batch, D_id + D_bert]

        x = user_emb * item_emb  # 可选：GMF，也可以改成 [user_emb ; item_emb]
        x = self.mlp_layers(x)
        return self.output(x).squeeze(-1)