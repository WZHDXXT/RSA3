import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import defaultdict
import random
import time

def load_implicit_data_with_negatives(
    filename,
    negative_sample_train=1,
    negative_sample_valid=10,
    train_ratio=0.85
):
    t0 = time.time()
    df = pd.read_csv(filename)
    df["label"] = 1

    all_users = df["user_id"].unique()
    all_items = df["item_id"].unique()
    user2id = {u: idx for idx, u in enumerate(all_users)}
    item2id = {i: idx for idx, i in enumerate(all_items)}

    df["user_id"] = df["user_id"].map(user2id)
    df["item_id"] = df["item_id"].map(item2id)

    grouped = df.groupby("user_id")["item_id"].apply(list)
    train_rows, val_rows = [], []

    all_item_set = set(range(len(item2id)))  # 预构建全集 item 集合

    for u, items in grouped.items():
        if len(items) < 2:
            continue  # 至少需要 1 个训练 1 个验证

        items = np.array(items)
        np.random.shuffle(items)
        split = int(len(items) * train_ratio)
        train_pos = items[:split]
        val_pos = items[split:]

        interacted = set(items)
        neg_pool = list(all_item_set - interacted)

        if len(neg_pool) < len(train_pos) * negative_sample_train:
            continue

        train_neg_sample_size = len(train_pos) * negative_sample_train
        train_neg = random.sample(neg_pool, train_neg_sample_size)
        val_neg = random.sample(neg_pool, negative_sample_valid)

        train_rows.extend([(u, i, 1) for i in train_pos])
        train_rows.extend([(u, i, 0) for i in train_neg])
        val_rows.extend([(u, i, 1) for i in val_pos])
        val_rows.extend([(u, i, 0) for i in val_neg])

    train_df = pd.DataFrame(train_rows, columns=["user_id", "item_id", "label"])
    val_df = pd.DataFrame(val_rows, columns=["user_id", "item_id", "label"])

    print("\n📌 train_df.head:")
    print(train_df.head(5))
    print("\n📌 val_df.head:")
    print(val_df.head(5))

    print(f"\n⏱️ Data loaded in {time.time() - t0:.2f}s")
    return train_df, val_df, len(item2id), len(user2id), user2id, item2id, df
