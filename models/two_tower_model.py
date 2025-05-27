import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(self, user_encoder, item_encoder):
        super().__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder  # May also be shared within user_encoder

    def forward(self, user_history_batch, item_input_batch):
        """
        user_history_batch: List[List[Dict]]  Each user's historical item inputs (multi-user batch)
        item_input_batch:   List[Dict]        Current item inputs to be scored
        Returns: scores [B], user_embs [B, D], item_embs [B, D]
        """
        device = next(self.parameters()).device

        # Encode users
        user_embs = []
        for history_input_dicts in user_history_batch:
            user_emb = self.user_encoder(history_input_dicts)
            user_embs.append(user_emb)
        user_embs = torch.cat(user_embs, dim=0)  # [B, 128]

        # Encode items
        item_embs = []
        for inputs in item_input_batch:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            emb = self.item_encoder(
                category=inputs['category'],
                store=inputs['store'],
                parent_asin=inputs['parent_asin'],
                text_embedding=inputs['text_embedding'],
                num_vec=inputs['num_vec'],
                image_vec=inputs['image_vec']
            )
            item_embs.append(emb)
        item_embs = torch.cat(item_embs, dim=0)  # [B, 128]

        # Dot product scores
        scores = torch.sum(user_embs * item_embs, dim=-1)  # [B]
        return scores, user_embs, item_embs
