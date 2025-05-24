# evaluate.py

import torch
from torch.utils.data import DataLoader
from utils.data_loader import RecDataset
from utils.metrics import recall_at_k
from utils.config import get_config
from models.two_tower_model import TwoTowerModel
import pickle
import pandas as pd
from tqdm import tqdm

@torch.no_grad()
def evaluate(model, dataloader, all_item_embeddings, device, k=10):
    model.eval()
    recalls = []
    preds = []
    targets = []

    for batch in tqdm(dataloader, desc="Evaluating"):
        seqs, _, labels = [x.to(device) for x in batch]
        user_embs = model.user_encoder(seqs)

        # 计算与所有item的相似度
        scores = torch.matmul(user_embs, all_item_embeddings.T)
        topk = torch.topk(scores, k=k, dim=-1).indices

        preds.extend(topk.cpu().tolist())
        targets.extend(labels.cpu().tolist())

    return recall_at_k(preds, targets, k)


def main():
    args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载元信息
    with open(args.item_meta_pkl, 'rb') as f:
        meta_dict = pickle.load(f)
    num_items = len(meta_dict)

    test_dataset = RecDataset(args.test_path, args.item_meta_pkl, args.max_seq_len)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = TwoTowerModel(num_items, args).to(device)
    model.load_state_dict(torch.load(args.save_path))

    # 获取所有item embedding
    item_ids = torch.arange(1, num_items + 1).to(device)
    item_embs = model.item_encoder(item_ids)

    recall = evaluate(model, test_loader, item_embs, device, k=args.top_k)
    print(f"Recall@{args.top_k}: {recall:.4f}")


if __name__ == '__main__':
    main()
