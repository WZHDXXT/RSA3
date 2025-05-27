import torch
from tqdm import tqdm
import pandas as pd
from models.item_encoder import ItemEncoder
from models.user_encoder_sequence import UserEncoder
from models.two_tower_model import TwoTowerModel
from collections import Counter
from utils.data_loader import TwoTowerTrainDataset, collate_fn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
import os
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)



if __name__ == "__main__":
    df_user = pd.read_csv('data/train.csv')
    df_user = df_user.sort_values(by=['user_id', 'timestamp'])
    user_histories = df_user.groupby('user_id')['item_id'].apply(list).to_dict()

    from sklearn.model_selection import train_test_split

    user_ids = list(user_histories.keys())
    train_users, val_users = train_test_split(user_ids, test_size=0.1, random_state=42)

    train_histories = {uid: user_histories[uid] for uid in train_users}
    val_histories = {uid: user_histories[uid] for uid in val_users}

    df = pd.read_csv('data/item_meta_processed.csv')
    num_categories = int(df['main_category_encoded'].max()) + 1
    num_stores = int(df['store_encoded'].max()) + 1
    num_items = int(df['item_id'].max()) + 1
    num_parent_asin = int(df['parent_asin_encoded'].max()) + 1
    print(num_categories, num_stores, num_parent_asin)
    item_encoder = ItemEncoder(
        num_categories=num_categories,
        num_stores=num_stores,
        num_parent_asin=num_parent_asin,
        text_embedding_dim=384
    )
    user_encoder = UserEncoder(num_items)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    item_freq = Counter([iid for ids in user_histories.values() for iid in ids])
    popular_items = [iid for iid, _ in item_freq.most_common(100)]  # top 100 popular items
    item_inputs = torch.load("preprocess/item_inputs.pt", map_location=device, weights_only=True)

    # invalid_users = [uid for uid, items in user_histories.items()
    #                  if all(i not in item_inputs for i in items)]
    # print(f"[Debug] Users with zero valid items: {len(invalid_users)}")

    from utils.data_loader import TwoTowerTrainDataset

    # Load training samples from tensor file
    # samples = torch.load("preprocess/train_samples_tensor.pt")
    # print(f"Loaded {len(samples)} training samples from tensor file.")
    # print('loading dataset')
    train_dataset = TwoTowerTrainDataset(train_histories, item_inputs, popular_items)
    val_dataset = TwoTowerTrainDataset(val_histories, item_inputs, popular_items)
    for i in range(5):
        sample = train_dataset[i]
        print(f"\n📦 Sample {i+1}")
        print(f"User ID:        {sample['user_id']}")
        print(f"History IDs:    {sample['history_ids']}")
        print(f"Positive Item:  {sample['pos_item_id']}")
        print(f"Negative Item:  {sample['neg_item_id']}")
    
    from functools import partial

    train_loader = DataLoader(
        train_dataset,
        batch_size=1024,
        shuffle=True,
        collate_fn=partial(collate_fn, item_inputs=item_inputs, device=device),
        # num_workers=4,
        # pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=partial(collate_fn, item_inputs=item_inputs, device=device),
        # num_workers=4,
        # pin_memory=True
    )


    model = TwoTowerModel(user_encoder, item_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    scaler = GradScaler(device='cuda' if torch.cuda.is_available() else 'cpu')
    for epoch in range(3):
        print(f"\n🔁 Epoch {epoch+1}")
        model.train()
        total_loss = 0.0

        # for input_ids, attention_mask, item_input_batch, labels in train_loader:
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            input_ids, attention_mask, item_input_batch, labels = batch
            labels = labels.to(device)

            with torch.no_grad():
                for item in item_input_batch:
                    for key in item:
                        item[key] = item[key].to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                scores, _, _ = model(input_ids, attention_mask, item_input_batch)
                loss = loss_fn(scores, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"✅ Epoch {epoch+1} Avg Loss: {total_loss / len(train_loader):.4f}")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, attention_mask, item_input_batch, labels in val_loader:
                labels = labels.to(device)
                for item in item_input_batch:
                    for key in item:
                        item[key] = item[key].to(device, non_blocking=True)
                scores, _, _ = model(input_ids, attention_mask, item_input_batch)
                loss = loss_fn(scores, labels)
                val_loss += loss.item()
        print(f"🧪 Val Avg Loss: {val_loss / len(val_loader):.4f}")

        # Save model
        torch.save(model.state_dict(), 'output/two_tower_model.pt')
        print("✅ Model saved to output/two_tower_model.pt")



# === Generate Top-10 predictions for test.csv ===
print("\n📤 Generating Top-10 predictions for test users...")

df_test = pd.read_csv('data/test.csv')
test_user_ids = df_test['user_id'].unique()
full_user_histories = {**train_histories, **val_histories}

model.eval()
all_item_ids = list(item_inputs.keys())
item_batch = {
    key: torch.stack([item_inputs[iid][key] for iid in all_item_ids]).to(device)
    for key in item_inputs[all_item_ids[0]]
}
with torch.no_grad():
    item_embeddings = model.item_encoder(**item_batch)  # [N, D]

recommendations = {}
with torch.no_grad():
    for user_id in test_user_ids:
        history = full_user_histories.get(user_id, [])
        if not history:
            continue  # skip cold start users

        input_ids = torch.tensor(history, dtype=torch.long).unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)
        user_emb = model.user_encoder(input_ids, attention_mask)  # [1, D]
        scores = torch.matmul(user_emb, item_embeddings.T).squeeze(0)  # [N]
        topk = torch.topk(scores, k=10)
        recommended_ids = [all_item_ids[i] for i in topk.indices.tolist()]
        recommendations[user_id] = recommended_ids

# Save predictions
output_df = pd.DataFrame([
    {'user_id': uid, 'item_id': ' '.join(map(str, items))}
    for uid, items in recommendations.items()
])
output_df.to_csv('output/submission.csv', index=False)
print("✅ Submission saved to output/submission.csv")
