import torch
import torch.nn as nn
import pandas as pd
from item_encoder import ItemEncoder
from user_encoder_sequence import UserEncoder

df = pd.read_csv('../data/item_meta_processed.csv')

# Use max() + 1 as the embedding vocabulary size (since encoding starts from 0)
num_categories = int(df['main_category_encoded'].max()) + 1
num_stores = int(df['store_encoded'].max()) + 1
num_parent_asin = int(df['parent_asin_encoded'].max()) + 1

# Initialize item encoder
item_encoder = ItemEncoder(
    num_categories=num_categories,
    num_stores=num_stores,
    num_parent_asin=num_parent_asin,
    text_embedding_dim=384
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

df_user = pd.read_csv('../data/train.csv')
df_user = df_user.sort_values(by=['user_id', 'timestamp'])
user_histories = df_user.groupby('user_id')['item_id'].apply(list).to_dict()

# Initialize user encoder (sharing item encoder)
user_encoder = UserEncoder(item_encoder)

# Select a user history from user_histories
user_id = 187732
item_ids = user_histories[user_id]
print(item_ids)
item_inputs = torch.load("item_inputs.pt", map_location=device)

# inputs = item_inputs[61600]
# print(inputs)
# Construct input: each item_id corresponds to its metadata input
history_input_dicts = [
    item_inputs[iid] for iid in item_ids if iid in item_inputs
]
# print(history_input_dicts)

# Call user encoder
user_embedding = user_encoder(history_input_dicts)  # shape: [1, 128]

print("User embedding shape:", user_embedding.shape)
print("User embedding:", user_embedding)
