import torch
import torch.nn as nn

class ItemEncoder(nn.Module):
    def __init__(self, num_categories, num_stores, num_parent_asin, text_embedding_dim=384):
        super(ItemEncoder, self).__init__()
        self.category_embed = nn.Embedding(num_categories + 1, 16)
        self.store_embed = nn.Embedding(num_stores + 1, 16)
        self.parent_asin_embed = nn.Embedding(num_parent_asin + 1, 16)
        self.text_fc = nn.Linear(text_embedding_dim, 64)  # text_embedding 替代 title
        self.output_fc = nn.Linear(16 + 16 + 16 + 64, 128)

    def forward(self, category, store, parent_asin, text_embedding):
        cat_emb = self.category_embed(category)
        store_emb = self.store_embed(store)
        parent_emb = self.parent_asin_embed(parent_asin)
        text_feat = self.text_fc(text_embedding)
        x = torch.cat([cat_emb, store_emb, parent_emb, text_feat], dim=-1)
        return self.output_fc(x)