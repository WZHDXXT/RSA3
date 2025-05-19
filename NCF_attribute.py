from transformers import BertTokenizer, BertModel
import ast

# ========= BERT 文本向量生成 =========
def generate_item_text_embeddings(item_meta_path, item2id, model_name="bert-base-uncased"):
    import torch
    import pandas as pd
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("loading model")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert = BertModel.from_pretrained(model_name).to(device)
    bert.eval()
    print("loading csv")
    meta_df = pd.read_csv(item_meta_path)
    meta_df = meta_df[meta_df["item_id"].isin(item2id.keys())].copy()

    def safe_parse(val):
        try:
            return ast.literal_eval(val)
        except:
            return []

    def build_text(row):
        parts = []
        if pd.notnull(row["main_category"]):
            print(row["main_category"])
            parts.append(f"Category: {row['main_category']}")
        if pd.notnull(row["title"]):
            parts.append(f"Title: {row['title']}")
        if pd.notnull(row["description"]) and row["description"] != "[]":
            parts.append(f"Description: {row['description']}")
        if pd.notnull(row["store"]):
            parts.append(f"Store: {row['store']}")
        features = safe_parse(row.get("features", "[]"))
        if features:
            parts.append("Features: " + ", ".join(features))
        details = row.get("details", "")
        if isinstance(details, str) and len(details) > 5:
            parts.append("Details: " + details)
        return " ".join(parts)

    meta_df["text"] = meta_df.apply(build_text, axis=1)

    embedding_dim = 768
    item_embedding_matrix = torch.zeros(len(item2id), embedding_dim)

    batch_size = 32
    texts = meta_df["text"].tolist()
    item_ids = meta_df["item_id"].tolist()
    print(f"📋 Total items to encode: {len(texts)}")
    print(f"🧾 Sample text[0]: {texts[0][:100]}")
    print(f"🆔 Sample item_id[0]: {item_ids[0]}")

    for start in tqdm(range(0, len(texts), batch_size)):
        end = min(start + batch_size, len(texts))
        batch_texts = texts[start:end]
        batch_item_ids = item_ids[start:end]

        # Tokenize batch
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = bert(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1).cpu()

        # 存入 embedding matrix
        for i, item_id in enumerate(batch_item_ids):
            if item_id in item2id:
                idx = item2id[item_id]
                item_embedding_matrix[idx] = embeddings[i]

    print(f"✅ BERT item_text_embedding shape: {item_embedding_matrix.shape}")
    return item_embedding_matrix

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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from model_bert import NeuMF
from data_load import load_implicit_data_with_negatives


# ========= 完整训练逻辑示例 =========
if __name__ == "__main__":
    file_name = "train.csv"
    item_meta_file = "item_meta.csv"
    test_file = "test.csv"

    train_df, val_df, num_items, num_users, user2id, item2id, full_df = load_implicit_data_with_negatives(file_name)

    item_vectors = generate_item_text_embeddings(item_meta_file, item2id)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_user = torch.tensor(train_df["user_id"].values, dtype=torch.long)
    train_item = torch.tensor(train_df["item_id"].values, dtype=torch.long)
    train_label = torch.tensor(train_df["label"].values, dtype=torch.float32)

    val_user = torch.tensor(val_df["user_id"].values, dtype=torch.long)
    val_item = torch.tensor(val_df["item_id"].values, dtype=torch.long)
    val_label = torch.tensor(val_df["label"].values, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(train_user, train_item, train_label), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_user, val_item, val_label), batch_size=1024)

    model = NeuMF(num_users + 1, num_items + 1, item_attr_dim=item_vectors.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    best_model = None
    best_loss = float("inf")
    for epoch in range(30):
        model.train()
        total_loss = 0
        for u, i, l in train_loader:
            u, i, l = u.to(device), i.to(device), l.to(device)
            item_attr = item_vectors[i].to(device)
            pred = model(u, i, item_attr)
            loss = criterion(pred, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for u, i, l in val_loader:
                u, i, l = u.to(device), i.to(device), l.to(device)
                item_attr = item_vectors[i].to(device)
                pred = model(u, i, item_attr)
                loss = criterion(pred, l)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        print(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model

    print("\n🔍 Evaluating Recall@10 on test set...")
    test_df = pd.read_csv(test_file)
    test_df = test_df[test_df["user_id"].isin(user2id)]
    test_df = test_df[test_df["item_id"].isin(item2id)]
    test_df["user_id"] = test_df["user_id"].map(user2id)
    test_df["item_id"] = test_df["item_id"].map(item2id)

    test_gt = defaultdict(set)
    for _, row in test_df.iterrows():
        test_gt[row["user_id"]].add(row["item_id"])

    recall_scores = []
    seen_items = full_df.groupby("user_id")["item_id"].apply(set).to_dict()

    best_model.eval()
    with torch.no_grad():
        for user_id in test_gt:
            seen = seen_items.get(user_id, set())
            candidate_items = list(set(range(num_items + 1)) - seen)

            item_tensor = torch.tensor(candidate_items, dtype=torch.long).to(device)
            user_tensor = torch.full_like(item_tensor, fill_value=user_id, dtype=torch.long).to(device)
            item_attr = item_vectors[item_tensor].to(device)

            scores = best_model(user_tensor, item_tensor, item_attr).squeeze(-1)
            top_items = torch.topk(scores, k=10).indices.cpu().numpy()
            top_items = [candidate_items[i] for i in top_items]

            hits = len(set(top_items) & test_gt[user_id])
            recall = hits / len(test_gt[user_id])
            recall_scores.append(recall)

    print(f"\n📊 Final Recall@10 on test.csv: {np.mean(recall_scores):.4f}")
