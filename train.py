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

def train(args):
    # Load metadata stats for embedding initialization
    num_items, num_categories, num_stores, num_parent_asin = get_metadata_stats(args.meta_path)

    # Build item embedding table for user encoder
    item_embedding = build_embedding_table(num_items, args.embedding_dim)

    # Initialize models
    user_encoder = UserEncoder(item_embedding).to(args.device)
    item_encoder = ItemEncoder(num_categories, num_stores, num_parent_asin).to(args.device)
    model = TwoTowerModel(user_encoder, item_encoder).to(args.device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Dataset and DataLoader
    train_dataset = RecDataset(args.train_path, args.meta_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # Training loop
    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in pbar:
            user_seq, item_features, labels = batch
            user_seq = user_seq.to(args.device)
            labels = labels.to(args.device)
            item_features = [feat.to(args.device) for feat in item_features]

            scores = model(user_seq, item_features)
            loss = F.binary_cross_entropy_with_logits(scores, labels.float())

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
    parser.add_argument('--model_dir', type=str, default='checkpoints/')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args)