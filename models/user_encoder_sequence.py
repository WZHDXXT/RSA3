import torch
import torch.nn as nn

class UserEncoder(nn.Module):
    def __init__(self, item_encoder, embedding_dim=128, max_seq_len=50, n_heads=4, n_layers=2):
        super(UserEncoder, self).__init__()
        self.item_encoder = item_encoder  # Shared item encoder
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len

        # Positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Optional projection layer (for [CLS] or mean pooling)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, history_input_dicts):
        device = next(self.parameters()).device
        embeddings = []

        # Step 1: Encode user history with item_encoder
        for inputs in history_input_dicts:
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = self.item_encoder(
                    category=inputs['category'],
                    store=inputs['store'],
                    parent_asin=inputs['parent_asin'],
                    text_embedding=inputs['text_embedding'],
                    num_vec=inputs['num_vec'],
                    image_vec=inputs['image_vec']
                )  # shape: [1, 128]
                embeddings.append(emb)

        if len(embeddings) == 0:
            return torch.zeros((1, self.embedding_dim), device=device)

        # Step 2: Concatenate into sequence
        seq = torch.cat(embeddings, dim=0)  # shape: [seq_len, 128]
        seq_len = seq.shape[0]
        seq = seq.unsqueeze(0)  # → [1, seq_len, 128]

        # Step 3: Add positional encoding
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0)  # [1, seq_len]
        pos_emb = self.position_embedding(position_ids)  # [1, seq_len, 128]
        seq = seq + pos_emb

        # Step 4: Transformer encoding
        transformer_output = self.transformer(seq)  # [1, seq_len, 128]

        # Step 5: Use the last token or mean pooling
        user_embedding = transformer_output.mean(dim=1)  # [1, 128]

        return self.output_layer(user_embedding)  # [1, 128]
