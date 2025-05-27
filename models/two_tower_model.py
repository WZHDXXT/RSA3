import torch
import torch.nn as nn

# class TwoTowerModel(nn.Module):
#     def __init__(self, user_encoder, item_encoder):
#         super().__init__()
#         self.user_encoder = user_encoder  # 输入 List[int]
#         self.item_encoder = item_encoder  # 输入 Dict[str, Tensor] 打包成 batch

#     def forward(self, user_history_batch, item_input_batch):
#         """
#         user_history_batch: List[List[int]] — 每个用户的历史 item_id 序列
#         item_input_batch: List[Dict[str, Tensor]] — 当前 item 的多模态特征
#         Returns:
#             scores: Tensor[B]
#             user_embs: Tensor[B, D]
#             item_embs: Tensor[B, D]
#         """
#         device = next(self.parameters()).device

#         # 用户编码：逐个送入 BERT 编码
#         user_embs = torch.cat([
#             self.user_encoder(history_ids).to(device) for history_ids in user_history_batch
#         ], dim=0)  # [B, D]

#         # Item 输入打包成 batch 张量
#         batch_inputs = {
#             key: torch.stack([item[key] for item in item_input_batch]).to(device)
#             for key in item_input_batch[0]
#         }

#         item_embs = self.item_encoder(**batch_inputs)  # [B, D]
#         item_embs = item_embs.squeeze(1)
#         # 相似度得分（点积）
#         scores = torch.sum(user_embs * item_embs, dim=-1)  # [B]
#         # print(scores)
#         return scores, user_embs, item_embs


class TwoTowerModel(nn.Module):
    def __init__(self, user_encoder, item_encoder):
        super().__init__()
        self.user_encoder = user_encoder
        self.item_encoder = item_encoder

    def forward(self, input_ids, attention_mask, item_input_batch):
        user_embs = self.user_encoder(input_ids, attention_mask)  # [B, D]

        batch_inputs = {
            key: torch.stack([item[key] for item in item_input_batch]).to(user_embs.device)
            for key in item_input_batch[0]
        }
        item_embs = self.item_encoder(**batch_inputs)  # [B, D]
    	
        item_embs = item_embs.squeeze(1)
        scores = torch.sum(user_embs * item_embs, dim=-1)  # [B]
        return scores, user_embs, item_embs


