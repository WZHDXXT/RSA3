# --- Added function for top-k prediction for submission ---
def predict_topk_for_submission(submission_path, train_path, meta_path, model_path, top_k=10, max_len=50, device='cpu'):
    import pandas as pd
    import torch
    from collections import defaultdict
    from models.user_encoder import UserEncoder
    from models.item_encoder import ItemEncoder
    from models.two_tower_model import TwoTowerModel

    # Step 1: Load metadata and model
    meta_df = pd.read_pickle(meta_path)
    num_items = len(meta_df)
    num_categories = meta_df['main_category'].max() + 1
    num_stores = meta_df['store'].max() + 1
    num_parent_asin = meta_df['parent_asin'].max() + 1

    item_features_tensor = {
        'category': torch.tensor(meta_df['main_category'].values, dtype=torch.long),
        'store': torch.tensor(meta_df['store'].values, dtype=torch.long),
        'parent_asin': torch.tensor(meta_df['parent_asin'].values, dtype=torch.long),
        'text_embedding': torch.tensor(meta_df['text_embedding'].tolist(), dtype=torch.float),
        'numeric_feats': torch.tensor(meta_df[['average_rating', 'rating_number', 'price']].values, dtype=torch.float)
    }

    item_encoder = ItemEncoder(num_categories, num_stores, num_parent_asin, text_embedding_dim=384).to(device)
    user_embedding = torch.nn.Embedding(meta_df['item_id'].max() + 2, 64).to(device)
    user_encoder = UserEncoder(user_embedding).to(device)
    model = TwoTowerModel(user_encoder, item_encoder).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Step 2: Build user sequences
    train_df = pd.read_csv(train_path).sort_values('timestamp')
    user_seq_dict = defaultdict(list)
    for _, row in train_df.iterrows():
        user_seq_dict[row['user_id']].append(row['item_id'])

    for uid in user_seq_dict:
        seq = user_seq_dict[uid][-max_len:]
        user_seq_dict[uid] = [0] * (max_len - len(seq)) + seq

    # Step 3: Load submission users and predict top-k
    submission_df = pd.read_csv(submission_path)
    user_ids = submission_df['user_id'].unique()

    all_item_vecs = item_encoder(
        item_features_tensor['category'].to(device),
        item_features_tensor['store'].to(device),
        item_features_tensor['parent_asin'].to(device),
        item_features_tensor['text_embedding'].to(device),
        item_features_tensor['numeric_feats'].to(device)
    )

    results = []

    for uid in user_ids:
        seq = user_seq_dict.get(uid, [0]*max_len)
        seq_tensor = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)
        user_vec = model.user_encoder(seq_tensor)  # (1, D)

        scores = torch.matmul(user_vec, all_item_vecs.T)  # (1, N)
        topk = torch.topk(scores, k=top_k, dim=-1).indices.squeeze(0)  # (top_k,)
        topk_items = [str(meta_df.iloc[i]['item_id']) for i in topk.tolist()]
        results.append({'user_id': uid, 'item_id': ','.join(topk_items)})

    result_df = pd.DataFrame(results)
    result_df['ID'] = result_df['user_id']
    result_df = result_df[['ID', 'user_id', 'item_id']]
    return result_df

if __name__ == "__main__":
    import argparse
    import torch
    args = argparse.ArgumentParser()
    args.add_argument('--submission_path', type=str, default='data/sample_submission.csv')
    args.add_argument('--train_path', type=str, default='data/train.csv')
    args.add_argument('--meta_path', type=str, default='data/processed_item_meta.pkl')
    args.add_argument('--model_path', type=str, default='output/two_tower_model.pt')
    args.add_argument('--output_path', type=str, default='output/submission.csv')
    args.add_argument('--top_k', type=int, default=10)
    args.add_argument('--max_len', type=int, default=50)
    args.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    config = args.parse_args()

    submission_df = predict_topk_for_submission(
        config.submission_path,
        config.train_path,
        config.meta_path,
        config.model_path,
        top_k=config.top_k,
        max_len=config.max_len,
        device=config.device
    )
    submission_df.to_csv(config.output_path, index=False)
    print(f"Saved prediction to {config.output_path}")