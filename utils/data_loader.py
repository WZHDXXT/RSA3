import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np

class RecDataset(Dataset):
    def __init__(self, csv_path, meta_pkl_path, max_seq_len=50):
        self.data = pd.read_csv(csv_path)
        with open(meta_pkl_path, 'rb') as f:
            self.item_meta = pickle.load(f)  # dict[item_id] = metadata

        self.max_seq_len = max_seq_len
        self.user_sequences = self._build_user_sequences()
        self.users = list(self.user_sequences.keys())

    def _build_user_sequences(self):
        user_history = defaultdict(list)
        for _, row in self.data.sort_values("timestamp").iterrows():
            user_history[row["user_id"]].append(row["item_id"])

        sequences = {}
        for user, items in user_history.items():
            if len(items) >= 2:
                sequences[user] = items
        return sequences

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user]
        input_seq = seq[:-1][-self.max_seq_len:]  # historical items
        target = seq[-1]  # predict next item

        padded_seq = [0] * (self.max_seq_len - len(input_seq)) + input_seq
        padded_seq = torch.tensor(padded_seq, dtype=torch.long)
        label = torch.tensor(target, dtype=torch.long)

        item_feat = self._get_item_features(target)

        return padded_seq, item_feat, label

    def _get_item_features(self, item_id):
        meta = self.item_meta.get(item_id, {})
        category = meta.get('main_category_id', 0)
        store = meta.get('store_id', 0)
        parent = meta.get('parent_asin_id', 0)
        return [torch.tensor(x, dtype=torch.long) for x in [category, store, parent]]


def build_embedding_table(num_items, dim):
    return torch.nn.Embedding(num_items + 1, dim, padding_idx=0)


def get_metadata_stats(meta_pkl_path):
    with open(meta_pkl_path, 'rb') as f:
        meta = pickle.load(f)
    category_set, store_set, parent_set = set(), set(), set()
    for m in meta.values():
        category_set.add(m.get('main_category_id', 0))
        store_set.add(m.get('store_id', 0))
        parent_set.add(m.get('parent_asin_id', 0))
    return len(meta), len(category_set), len(store_set), len(parent_set)
