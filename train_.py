import torch
import random
import pandas as pd
from models.item_encoder import ItemEncoder
from models.user_encoder_sequence import UserEncoder
from models.two_tower_model import TwoTowerModel
from collections import Counter
from utils.data_loader import TwoTowerTrainDataset, collate_fn

from torch.utils.data import DataLoader


df_user = pd.read_csv('data/train.csv')
df_user = df_user.sort_values(by=['user_id', 'timestamp'])
user_histories = df_user.groupby('user_id')['item_id'].apply(list).to_dict()


df = pd.read_csv('data/item_meta_processed.csv')
num_categories = int(df['main_category_encoded'].max()) + 1
num_stores = int(df['store_encoded'].max()) + 1
num_parent_asin = int(df['parent_asin_encoded'].max()) + 1
print(num_categories, num_stores, num_parent_asin)
item_encoder = ItemEncoder(
    num_categories=num_categories,
    num_stores=num_stores,
    num_parent_asin=num_parent_asin,
    text_embedding_dim=384
)
user_encoder = UserEncoder(item_encoder)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

item_freq = Counter([iid for ids in user_histories.values() for iid in ids])
popular_items = [iid for iid, _ in item_freq.most_common(100)]  # top 100 popular items
item_inputs=torch.load("data/item_inputs.pt", map_location=device, weights_only=True)


# invalid_users = [uid for uid, items in user_histories.items()
#                  if all(i not in item_inputs for i in items)]
# print(f"[Debug] Users with zero valid items: {len(invalid_users)}")

import json

from utils.data_loader import TwoTowerTrainDataset


# Load JSONL training samples
import json

with open("preprocess/train_samples.json", "r") as f:
    samples = [json.loads(line) for line in f]
print('loading dataset')
train_dataset = TwoTowerTrainDataset(samples, item_inputs)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda x: collate_fn(x, item_inputs, item_encoder, user_encoder, device)
)


# for batch_idx, (user_history_batch, item_input_batch, labels) in enumerate(train_loader):
#     print(f"\n📦 Batch {batch_idx + 1}")
#     print(f"User history batch size: {len(user_history_batch)}")  # should be 2×batch_size (1 positive + 1 negative)
#     print(f"Item input batch size: {len(item_input_batch)}")
#     print(f"Labels: {labels.tolist()}")

#     # Print the number of items in the first user's history
#     print("\n🔍 Number of items in the first user's history:", len(user_history_batch[0]))
#     print("Sample item keys:", list(user_history_batch[0][0].keys()))  # e.g., ['category', 'store', ...]

#     # Print some dimensions of one item
#     print("\n🧩 First item_input:")
#     example_item = item_input_batch[0]
#     for k, v in example_item.items():
#         print(f" - {k}: shape {tuple(v.shape)}")

#     # Limit to look at only 1-2 batches
#     if batch_idx >= 1:
#         break
print('loading model')

model = TwoTowerModel(user_encoder, item_encoder).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(3):
    print(f'training with epoch {epoch}')
    model.train()
    for user_histories_batch, item_inputs_batch, labels in train_loader:
        scores, _, _ = model(user_histories_batch, item_inputs_batch)
        loss = loss_fn(scores, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")
