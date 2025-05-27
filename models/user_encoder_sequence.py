import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

# class UserEncoder(nn.Module):
#     def __init__(self, num_items, embedding_dim=128, max_seq_len=30):
#         super(UserEncoder, self).__init__()

#         # 初始化一个简单的 BERT 配置
#         config = BertConfig(
#             vocab_size=num_items,
#             hidden_size=embedding_dim,
#             num_hidden_layers=2,
#             num_attention_heads=4,
#             intermediate_size=embedding_dim * 4,
#             max_position_embeddings=max_seq_len,
#             pad_token_id=0
#         )
#         self.bert = BertModel(config)

#     def forward(self, history_item_ids):
#         """
#         history_item_ids: List[int] — 用户历史 item_id 列表
#         Returns: Tensor [1, embedding_dim]
#         """
#         if len(history_item_ids) == 0:
#             return torch.zeros((1, self.bert.config.hidden_size), device=next(self.parameters()).device)

#         input_ids = torch.tensor(history_item_ids, dtype=torch.long, device=next(self.parameters()).device).unsqueeze(0)  # [1, seq_len]
#         attention_mask = torch.ones_like(input_ids)  # [1, seq_len]

#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden_size]

#         # 平均池化
#         user_emb = last_hidden.mean(dim=1)  # [1, hidden_size]
#         return user_emb


class UserEncoder(nn.Module):
    def __init__(self, num_items, embedding_dim=128, max_seq_len=30):
        super().__init__()
        config = BertConfig(
            vocab_size=num_items,
            hidden_size=embedding_dim,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=embedding_dim * 4,
            max_position_embeddings=max_seq_len,
            pad_token_id=0
        )
        self.bert = BertModel(config)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state.mean(dim=1)  # [B, D]
