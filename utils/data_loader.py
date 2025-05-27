from tqdm import tqdm
from torch.utils.data import Dataset
import random
import torch
import json

import torch
from torch.utils.data import Dataset
import random

from torch.utils.data import Dataset
import random

class TwoTowerTrainDataset(Dataset):
    def __init__(self, user_histories, item_inputs, popular_items, num_negatives=1, max_seq_len=30):
        """
        user_histories: Dict[int, List[int]] — 每个用户的 item_id 序列
        item_inputs: Dict[int, Dict[str, Tensor]] — 有效的 item 特征
        popular_items: List[int] — 常见热门 item，用于补充不足的历史
        """
        self.item_inputs = item_inputs
        self.all_item_ids = list(item_inputs.keys())
        self.popular_items = [iid for iid in popular_items if iid in item_inputs]
        self.num_negatives = num_negatives
        self.max_seq_len = max_seq_len

        self.user_histories = {}
        for uid, item_ids in user_histories.items():
            filtered = [iid for iid in item_ids if iid in item_inputs]

            # 若少于 2 条，补热门 item（避免重复）
            if len(filtered) < 2:
                needed = 2 - len(filtered)
                supplement = [iid for iid in self.popular_items if iid not in filtered]
                filtered += supplement[:needed]

            if len(filtered) >= 2:
                self.user_histories[uid] = filtered

        self.user_ids = list(self.user_histories.keys())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        item_ids = self.user_histories[user_id]

        pos_item_id = item_ids[-1]
        history_ids = item_ids[:-1][-self.max_seq_len:]

        exclude_set = set(item_ids)
        candidates = [iid for iid in self.all_item_ids if iid not in exclude_set]
        if not candidates:
            neg_item_id = random.choice(self.all_item_ids)
        else:
            neg_item_id = random.choice(candidates)

        return {
            'user_id': user_id,
            'history_ids': history_ids,
            'pos_item_id': pos_item_id,
            'neg_item_id': neg_item_id
        }


def collate_fn(batch, item_inputs, device='cpu', max_seq_len=30):
    user_history_batch = []
    item_input_batch = []
    labels = []

    for sample in batch:
        history_ids = sample['history_ids'][-max_seq_len:]
        pos_id = sample['pos_item_id']
        neg_id = sample['neg_item_id']

        if pos_id not in item_inputs or neg_id not in item_inputs:
            continue

        user_history_batch.append(history_ids)
        item_input_batch.append(item_inputs[pos_id])
        labels.append(1.0)

        user_history_batch.append(history_ids)
        item_input_batch.append(item_inputs[neg_id])
        labels.append(0.0)

    if len(labels) == 0:
        return torch.empty(0), torch.empty(0), [], torch.tensor([], dtype=torch.float32, device=device)

    # Padding for BERT
    max_len = max(len(seq) for seq in user_history_batch)
    input_ids = torch.zeros(len(user_history_batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)

    for i, seq in enumerate(user_history_batch):
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, :len(seq)] = 1

    label_tensor = torch.tensor(labels, dtype=torch.float32, device=device)

    return input_ids.to(device), attention_mask.to(device), item_input_batch, label_tensor
