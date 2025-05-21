from model1 import Bert4Rec
from data_load import MakeSequenceDataSet
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# === Load train.csv and build user2id mapping ===
df = pd.read_csv("train.csv")
df["label"] = 1

all_users = df["user_id"].unique()
user2id = {u: i for i, u in enumerate(all_users)}

# === Sort by timestamp and build user→item sequences ===
df = df.sort_values(["user_id", "timestamp"])

user_sequences = defaultdict(list)
for _, row in df.iterrows():
    uid = row["user_id"]
    user_sequences[uid].append(row["item_id"])

# === Build user list aligned with user2id index order ===
sorted_user_ids = sorted(user2id, key=lambda u: user2id[u])

# === Convert each user's sequence to tensor ===
max_len = 20
seq_tensors = []
for uid in sorted_user_ids:
    item_seq = user_sequences[uid][-max_len:]
    seq_tensor = torch.tensor(item_seq, dtype=torch.long)
    seq_tensors.append(seq_tensor)

padded_seqs = pad_sequence(seq_tensors, batch_first=True, padding_value=0).to(device)

# === Load trained Bert4Rec model ===
bert_model = Bert4Rec(
    max_seq_length=20,
    vocab_size=df["item_id"].nunique(),  # or pass vocab_size=num_items
    bert_num_blocks=2,
    bert_num_heads=2,
    hidden_size=64,
    bert_dropout=0.1
)
bert_model.load_state_dict(torch.load("bert4rec_best.pt"))
bert_model.eval()
bert_model.to(device)

# === Batch generate user embeddings ===
batch_size = 64
user_embs = []

with torch.no_grad():
    for i in range(0, len(padded_seqs), batch_size):
        batch = padded_seqs[i:i+batch_size]
        attention_mask = (batch != 0).float().to(device)
        token_type_ids = torch.zeros_like(batch)

        outputs = bert_model.bert(
            input_ids=batch,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        ).last_hidden_state

        pooled = outputs.mean(dim=1)  # [batch, hidden]
        user_embs.append(pooled.cpu())

user_embs = torch.cat(user_embs, dim=0)  # [num_users, hidden]
np.save("user_bert4rec_embeddings.npy", user_embs.numpy())
print("✅ Saved user_bert4rec_embeddings.npy")
