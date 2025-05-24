import torch
import torch.nn as nn

class UserEncoder(nn.Module):
    def __init__(self, item_embedding):
        super(UserEncoder, self).__init__()
        self.item_embedding = item_embedding
        self.gru = nn.GRU(input_size=item_embedding.embedding_dim, hidden_size=128, batch_first=True)

    def forward(self, item_seq):
        if item_seq.dim() == 3 and item_seq.size(-1) == 1:
            item_seq = item_seq.squeeze(-1)
        embed_seq = self.item_embedding(item_seq)  # (B, T, E)
        _, h = self.gru(embed_seq)  # h: (1, B, H)
        return h.squeeze(0)  # (B, H)