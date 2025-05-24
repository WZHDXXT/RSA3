import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.user_encoder import UserEncoder
from models.item_encoder import ItemEncoder
from models.two_tower_model import TwoTowerModel
from utils.data_loader import RecDataset, build_embedding_table, get_metadata_stats
from utils.metrics import recall_at_k
from tqdm import tqdm
import argparse
import os
import pandas as pd

def train(args):
    # Load metadata stats for embedding initialization
    meta_df = pd.read_pickle(args.meta_path)
    num_items = len(meta_df)
    # num_categories = meta_df['main_category'].nunique()
    num_categories = meta_df['main_category'].max() + 1
    num_stores = meta_df['store'].max() + 1
    num_parent_asin = meta_df['parent_asin'].max() + 1

    print("[DEBUG] Number of items:", num_items)
    print("[DEBUG] Number of categories:", num_categories)
    print("[DEBUG] Number of stores:", num_stores)
    print("[DEBUG] Number of parent_asin:", num_parent_asin)

    # Read train.csv to get max item_id from both train and meta
    train_df = pd.read_csv(args.train_path)
    max_item_id = max(train_df['item_id'].max(), meta_df['item_id'].max())
    item_embedding = build_embedding_table(max_item_id + 1, args.embedding_dim)

    # Dataset and DataLoader
    train_dataset = RecDataset(args.train_path, args.meta_path)

    print("[DEBUG] Training dataset size:", len(train_dataset))
    print("[DEBUG] Sample from dataset:")
    sample = train_dataset[0]
    if isinstance(sample, list):  # multiple samples (e.g. pos + neg)
        for i, s in enumerate(sample):
            print(f"  Sample {i}:")
            print("    Padded sequence:", s[0])
            print("    Item features:", s[1])
            print("    Label:", s[2])
    else:
        print("  Padded sequence:", sample[0])
        print("  Item features:", sample[1])
        print("  Label:", sample[2])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Initialize models
    user_encoder = UserEncoder(item_embedding).to(args.device)
    item_encoder = ItemEncoder(num_categories, num_stores, num_parent_asin, text_embedding_dim=384).to(args.device)
    model = TwoTowerModel(user_encoder, item_encoder).to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            user_seq, item_features, labels, item_ids = zip(*batch)
            if isinstance(item_features[0], (list, tuple)):
                # Flatten batched positive and negative samples
                user_seq = torch.cat(user_seq, dim=0)
                labels = torch.cat(labels, dim=0)
                item_features = [torch.cat([f[i] for f in item_features], dim=0) for i in range(len(item_features[0]))]
                # item_ids is a tuple of tuples, flatten it and convert tensors to Python scalars
                flat_item_ids = []
                for group in item_ids:
                    for t in group:
                        if isinstance(t, torch.Tensor):
                            flat_item_ids.append(t.item())
                        else:
                            flat_item_ids.append(t)
                item_ids = tuple(flat_item_ids)
            else:
                user_seq = torch.stack(user_seq)
                labels = torch.stack(labels)
            user_seq = user_seq.to(args.device)
            labels = labels.to(args.device)
            item_features = [feat.cuda() if args.device.startswith('cuda') else feat.cpu() for feat in item_features]
            category, store, parent_asin, text_embedding = item_features

            # print(f"[DEBUG] category max: {category.max().item()}, embedding size: {num_categories}")
            # print(f"[DEBUG] store max: {store.max().item()}, embedding size: {num_stores}")
            # print(f"[DEBUG] parent_asin max: {parent_asin.max().item()}, embedding size: {num_parent_asin}")
            # print("[DEBUG] item_ids sample:", item_ids[:5])

            scores = model(user_seq, *item_features)
            # print("[DEBUG] scores sample:", scores[:5].detach().cpu().numpy())
            loss = F.binary_cross_entropy_with_logits(scores, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.model_dir, "two_tower_model.pt"))
    print("Training complete. Model saved.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--meta_path', type=str, default='data/processed_item_meta.pkl')
    parser.add_argument('--model_dir', type=str, default='models/')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)