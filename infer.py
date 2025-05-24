# infer.py

import torch
from torch.utils.data import DataLoader
from utils.data_loader import RecDataset
from utils.config import get_config
from models.two_tower_model import TwoTowerModel
import pickle
import pandas as pd
from tqdm import tqdm
import os

@torch.no_grad()
def generate_recommendations(model, dataloader, all_item_embeddings, device, k=10):
    model.eval()
    user_ids = []
    recommendations = []

    for batch in tqdm(dataloader, desc="Inferencing"):
        seqs, _, _ = [x.to(device) for x in batch]
        batch_user_ids = dataloader.dataset.users
        user_embs = model.user_encoder(seqs)

        scores = torch.matmul(user_embs, all_item_embeddings.T)
        topk = torch.topk(scores, k=k, dim=-1).indices

        user_ids.extend(batch_user_ids)
        recommendations.extend([" ".join(map(str, rec)) for rec in topk.cpu().tolist()])

    return user_ids, recommendations


def main():
    args = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载item元信息
    with open(args.item_meta_pkl, 'rb') as f:
        meta_dict = pickle.load(f)
    num_items = len(meta_dict)

    # 预测用户列表（来自test.csv或sample_submission）
    infer_dataset = RecDataset(args.test_path, args.item_meta_pkl, args.max_seq_len)
    infer_loader = DataLoader(infer_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 加载模型
    model = TwoTowerModel(num_items, args).to(device)
    model.load_state_dict(torch.load(args.save_path))

    # 所有item的embedding
    item_ids = torch.arange(1, num_items + 1).to(device)
    item_embs = model.item_encoder(item_ids)

    # 推理并保存结果
    user_ids, recs = generate_recommendations(model, infer_loader, item_embs, device, k=args.top_k)
    submission_df = pd.DataFrame({"user_id": user_ids, "item_id": recs})
    submission_df['ID'] = submission_df['user_id']
    submission_df = submission_df[['ID', 'user_id', 'item_id']]
    os.makedirs("output", exist_ok=True)
    submission_df.to_csv("output/submission.csv", index=False)
    print("Saved to output/submission.csv")


if __name__ == '__main__':
    main()