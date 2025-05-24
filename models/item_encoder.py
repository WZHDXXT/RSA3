import torch
import torch.nn as nn

class ItemEncoder(nn.Module):
    def __init__(self, num_categories, num_stores, num_parent_asin):
        super(ItemEncoder, self).__init__()
        self.category_embed = nn.Embedding(num_categories, 16)
        self.store_embed = nn.Embedding(num_stores, 16)
        self.parent_asin_embed = nn.Embedding(num_parent_asin, 16)
        self.numeric_fc = nn.Linear(3, 16)  # average_rating, rating_number, price
        self.title_fc = nn.Linear(384, 64)  # 假设使用的Sentence-BERT输出维度为384
        self.output_fc = nn.Linear(16 + 16 + 16 + 16 + 64, 128)

    def forward(self, category, store, parent_asin, numeric_features, title_embedding):
        cat_emb = self.category_embed(category)
        store_emb = self.store_embed(store)
        parent_asin_emb = self.parent_asin_embed(parent_asin)
        num_feat = self.numeric_fc(numeric_features)
        title_emb = self.title_fc(title_embedding)
        x = torch.cat([cat_emb, store_emb, parent_asin_emb, num_feat, title_emb], dim=-1)
        return self.output_fc(x)