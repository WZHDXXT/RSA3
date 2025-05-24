# evaluate.py
import torch
from torch.utils.data import DataLoader
from utils.metrics import recall_at_k
from tqdm import tqdm

@torch.no_grad()
def evaluate_topk_recall(model, dataloader, item_embs, device, top_k=10):
    model.eval()
    total_preds = []
    total_targets = []

    for batch in tqdm(dataloader, desc=f"Evaluating Recall@{top_k}"):
        user_seq = torch.stack([x[0] for x in batch]).to(device)       # (B, T)
        target_items = torch.tensor([x[3].item() for x in batch]).to(device)  # (B,)

        user_vec = model.user_encoder(user_seq)                        # (B, D)
        scores = torch.matmul(user_vec, item_embs.T)                   # (B, num_items)

        topk_items = torch.topk(scores, k=top_k, dim=-1).indices       # (B, K)

        total_preds.extend(topk_items.cpu().tolist())
        total_targets.extend(target_items.cpu().tolist())

    recall = recall_at_k(total_preds, total_targets, k=top_k)
    print(f"[Evaluation] Recall@{top_k}: {recall:.4f}")
    return recall


def main():
    import argparse
    import pandas as pd
    from models.user_encoder import UserEncoder
    from models.item_encoder import ItemEncoder
    from models.two_tower_model import TwoTowerModel
    from utils.data_loader import RecDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--meta_path', type=str, default='data/processed_item_meta.pkl')
    parser.add_argument('--model_path', type=str, default='models/two_tower_model.pt')
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load metadata
    meta_df = pd.read_pickle(args.meta_path)
    test_df = pd.read_csv(args.test_path)
    max_item_id = max(meta_df['item_id'].max(), test_df['item_id'].max())
    num_items = max_item_id + 1
    num_categories = meta_df['main_category'].max() + 1
    num_stores = meta_df['store'].max() + 1
    num_parent_asin = meta_df['parent_asin'].max() + 1

    # Initialize encoders and model
    item_embedding = torch.nn.Embedding(num_items + 1, args.embedding_dim).to(device)
    user_encoder = UserEncoder(item_embedding).to(device)
    item_encoder = ItemEncoder(num_categories, num_stores, num_parent_asin, text_embedding_dim=384).to(device)
    model = TwoTowerModel(user_encoder, item_encoder).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # Build item embeddings
    category = torch.tensor(meta_df['main_category'].values, dtype=torch.long).to(device)
    store = torch.tensor(meta_df['store'].values, dtype=torch.long).to(device)
    parent_asin = torch.tensor(meta_df['parent_asin'].values, dtype=torch.long).to(device)
    text_embedding = torch.tensor(meta_df['text_embedding'].tolist(), dtype=torch.float).to(device)
    item_embs = model.item_encoder(category, store, parent_asin, text_embedding)

    # Prepare test dataloader (only positive samples)
    test_dataset = RecDataset(args.test_path, args.meta_path, mode='eval')  # <-- mode='eval'
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Evaluate
    evaluate_topk_recall(model, test_loader, item_embs, device, top_k=args.top_k)


if __name__ == '__main__':
    main()
