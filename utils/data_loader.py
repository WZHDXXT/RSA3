import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from collections import defaultdict
import numpy as np

class RecDataset(Dataset):
    def __init__(self, csv_path, meta_pkl_path, max_seq_len=50, num_negatives=1, mode='train'):
        self.data = pd.read_csv(csv_path)
        with open(meta_pkl_path, 'rb') as f:
            raw_meta = pickle.load(f)
            if isinstance(raw_meta, pd.DataFrame):
                self.item_meta = raw_meta.set_index('item_id').to_dict(orient='index')
            else:
                self.item_meta = raw_meta
        # print(self.item_meta)
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.num_negatives = num_negatives
        self.user_sequences = self._build_user_sequences()
        self.users = list(self.user_sequences.keys())

    def _build_user_sequences(self):
        user_history = defaultdict(list)
        for _, row in self.data.sort_values("timestamp").iterrows():
            user_history[row["user_id"]].append(row["item_id"])

        sequences = {}
        for user, items in user_history.items():
            if len(items) >= 1:
                sequences[user] = items
        return sequences

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_sequences[user]
        input_seq = seq[:-1][-self.max_seq_len:]
        target_pos = seq[-1]  # 正样本

        padded_seq = [0] * (self.max_seq_len - len(input_seq)) + input_seq
        padded_seq = torch.tensor(padded_seq, dtype=torch.long)

        pos_feat = self._get_item_features(target_pos)
        pos_label = torch.tensor(1.0, dtype=torch.float)

        all_items = set(self.item_meta.keys())
        user_items = set(seq)
        negative_candidates = list(all_items - user_items)

        neg_samples = []
        for _ in range(self.num_negatives):
            target_neg = np.random.choice(negative_candidates)
            neg_feat = self._get_item_features(target_neg)
            neg_label = torch.tensor(0.0, dtype=torch.float)
            neg_samples.append((padded_seq, neg_feat, neg_label, target_neg))

        samples = [(padded_seq, pos_feat, pos_label, target_pos)] + neg_samples
        if self.mode == 'eval':
            return samples[0]  # 只返回正样本
        return samples

    def _get_item_features(self, item_id):
        meta = self.item_meta.get(item_id, {})
        category = meta.get('main_category', 0)
        store = meta.get('store', 0)
        parent = meta.get('parent_asin', 0)
        text_embed = meta.get('text_embedding', [0.0] * 384)
        text_embed_tensor = torch.tensor(text_embed, dtype=torch.float)
        return [
            torch.tensor(category, dtype=torch.long),
            torch.tensor(store, dtype=torch.long),
            torch.tensor(parent, dtype=torch.long),
            text_embed_tensor
        ]


def build_embedding_table(num_items, dim):
    return torch.nn.Embedding(num_items + 1, dim, padding_idx=0)



def get_metadata_stats(meta_pkl_path):
    with open(meta_pkl_path, 'rb') as f:
        meta = pickle.load(f)
    category_set, store_set, parent_set = set(), set(), set()
    for m in meta.values():
        category_set.add(m.get('main_category', 0))
        store_set.add(m.get('store', 0))
        parent_set.add(m.get('parent_asin', 0))
    return len(meta), len(category_set), len(store_set), len(parent_set)


# Main function for testing dataset loading
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='../data/train.csv')
    parser.add_argument('--meta_pkl_path', type=str, default='../data/processed_item_meta.pkl')
    args = parser.parse_args()

    print("Loading dataset...")
    dataset = RecDataset(args.csv_path, args.meta_pkl_path)
    print(f"Number of users: {len(dataset)}")
    sample = dataset[0]
    print(len(sample))
    print("Sample data:")
    print("  Padded sequence:", sample[0])
    print("  Item features:", sample[1])
    print("  Label:", sample[2])
    # print("Sample metadata for target item:", dataset.item_meta.get(int(sample[2]), {}))
