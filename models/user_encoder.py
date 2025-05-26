import torch
import torch.nn as nn

class UserEncoder(nn.Module):
    def __init__(self, item_encoder):
        super(UserEncoder, self).__init__()
        self.item_encoder = item_encoder  # shared encoder

    def forward(self, history_input_dicts):
        """
        history_input_dicts: List[Dict]  # history item information
        Returns: Tensor [1, embedding_dim]
        """
        device = next(self.item_encoder.parameters()).device
        embeddings = []

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
            return torch.zeros((1, 128), device=device)
        else:
            return torch.mean(torch.stack(embeddings, dim=0), dim=0)  # shape: [1, 128]
