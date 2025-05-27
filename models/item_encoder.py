import torch
import torch.nn as nn


    # return {
    #     'category': category,
    #     'store': store,
    #     'parent_asin': parent_asin,
    #     'text_embedding': text_vec,
    #     'num_vec': num_vec,
    #     'image_vec': image_vec
    # }

# class ItemEncoder(nn.Module):
#     def __init__(self, num_categories, num_stores, num_parent_asin, text_embedding_dim=384):
#         super(ItemEncoder, self).__init__()
#         self.category_embed = nn.Embedding(num_categories + 1, 16) # category to 16
#         self.store_embed = nn.Embedding(num_stores + 1, 16) # store to 16
#         self.parent_asin_embed = nn.Embedding(num_parent_asin + 1, 16) # parent_asin to 16
#         self.text_fc = nn.Linear(text_embedding_dim, 64)  # text 64
#         self.num_fc = nn.Linear(3, 16) # num_vec 16
#         self.image_fc = nn.Linear(512, 32) # image_vec 32
#         self.output_fc = nn.Linear(160, 128) # 16*3 + 64 + 16 + 32 =160

#     def forward(self, category, store, parent_asin, text_embedding, num_vec, image_vec):
#         cat_emb = self.category_embed(category)
#         store_emb = self.store_embed(store)
#         parent_emb = self.parent_asin_embed(parent_asin)
#         text_feat = self.text_fc(text_embedding)
#         num_feat = self.num_fc(num_vec)
#         img_feat = self.image_fc(image_vec)
#         x = torch.cat([cat_emb, store_emb, parent_emb, text_feat, num_feat, img_feat], dim=-1)
#         return self.output_fc(x)



import torch.nn as nn
import torch.nn.functional as F

class ItemEncoder(nn.Module):
    def __init__(self, num_categories, num_stores, num_parent_asin, text_embedding_dim=384, embedding_dim=128):
        super(ItemEncoder, self).__init__()
        self.category_embed = nn.Embedding(num_categories + 1, 16)
        self.store_embed = nn.Embedding(num_stores + 1, 16)
        self.parent_asin_embed = nn.Embedding(num_parent_asin + 1, 16)

        self.text_fc = nn.Sequential(
            nn.Linear(text_embedding_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        self.num_fc = nn.Sequential(
            nn.Linear(3, 16),
            nn.LayerNorm(16),
            nn.ReLU()
        )

        self.image_fc = nn.Sequential(
            nn.Linear(768, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # Fusion MLP layer
        self.fusion_mlp = nn.Sequential(
            nn.Linear(160, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, category, store, parent_asin, text_embedding, num_vec, image_vec):
        device = next(self.parameters()).device
        category = category.to(device)
        store = store.to(device)
        parent_asin = parent_asin.to(device)
        text_embedding = text_embedding.to(device)
        num_vec = num_vec.to(device).float()
        image_vec = image_vec.to(device).float()

        cat_emb = self.category_embed(category)
        store_emb = self.store_embed(store)
        parent_emb = self.parent_asin_embed(parent_asin)
        text_feat = self.text_fc(text_embedding)
        num_feat = self.num_fc(num_vec)
        img_feat = self.image_fc(image_vec)

        x = torch.cat([cat_emb, store_emb, parent_emb, text_feat, num_feat, img_feat], dim=-1)
        out = self.fusion_mlp(x)
        return F.normalize(out, dim=-1)  # Normalize to align with user embedding for dot-product scoring
