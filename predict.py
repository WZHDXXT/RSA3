import torch
import pandas as pd
from collections import Counter
from models.item_encoder import ItemEncoder
from models.user_encoder_sequence import UserEncoder
from models.two_tower_model import TwoTowerModel

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    df_user = pd.read_csv('data/train.csv')
    df_user = df_user.sort_values(by=['user_id', 'timestamp'])
    user_histories = df_user.groupby('user_id')['item_id'].apply(list).to_dict()

    from sklearn.model_selection import train_test_split
    user_ids = list(user_histories.keys())
    train_users, val_users = train_test_split(user_ids, test_size=0.1, random_state=42)
    train_histories = {uid: user_histories[uid] for uid in train_users}
    val_histories = {uid: user_histories[uid] for uid in val_users}

    df_item = pd.read_csv('data/item_meta_processed.csv')
    num_categories = int(df_item['main_category_encoded'].max()) + 1
    num_stores = int(df_item['store_encoded'].max()) + 1
    num_parent_asin = int(df_item['parent_asin_encoded'].max()) + 1

    item_encoder = ItemEncoder(num_categories, num_stores, num_parent_asin, text_embedding_dim=384)
    user_encoder = UserEncoder(int(df_item['item_id'].max()) + 1)

    item_inputs = torch.load("preprocess/item_inputs.pt", map_location=device, weights_only=True)

    model = TwoTowerModel(user_encoder, item_encoder).to(device)
    model.load_state_dict(torch.load("output/two_tower_model.pt", map_location=device))
    model.eval()

    df_test = pd.read_csv('data/test.csv')
    test_user_ids = df_test['user_id'].unique()
    full_user_histories = {**train_histories, **val_histories}

    all_item_ids = list(item_inputs.keys())
    item_batch = {
        key: torch.stack([item_inputs[iid][key].squeeze() for iid in all_item_ids]).to(device)
        for key in item_inputs[all_item_ids[0]]
    }

    with torch.no_grad():
        item_embeddings = model.item_encoder(**item_batch)

    max_seq_len = 30  # 固定长度

    recommendations = {}
    with torch.no_grad():
        for user_id in test_user_ids:
            history = full_user_histories.get(user_id, [])
            if not history:
                continue  # cold start skip

            # 截断或补零到 max_seq_len
            if len(history) > max_seq_len:
                history = history[-max_seq_len:]  # 保留最近的
            else:
                history = [0] * (max_seq_len - len(history)) + history  # 前补0

            input_ids = torch.tensor(history, dtype=torch.long).unsqueeze(0).to(device)  # [1, max_seq_len]
            attention_mask = (input_ids != 0).long().to(device)  # [1, max_seq_len]

            user_emb = model.user_encoder(input_ids, attention_mask)  # [1, embedding_dim]
            scores = torch.matmul(user_emb, item_embeddings.T).squeeze(0)  # [num_items]

            topk = torch.topk(scores, k=10)
            recommended_ids = [all_item_ids[i] for i in topk.indices.tolist()]
            recommendations[user_id] = recommended_ids

    output_df = pd.DataFrame([
        {'user_id': uid, 'item_id': ' '.join(map(str, items))}
        for uid, items in recommendations.items()
    ])
    output_df.to_csv('output/submission.csv', index=False)
    print("✅ Submission saved to output/submission.csv")

if __name__ == "__main__":
    main()


