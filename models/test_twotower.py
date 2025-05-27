import torch
import random
import pandas as pd
from item_encoder import ItemEncoder
from user_encoder_sequence import UserEncoder
from two_tower_model import TwoTowerModel
from collections import Counter

def test_two_tower_model(user_id, user_histories, item_inputs, model, popular_items, device='cpu'):
    """
    测试 TwoTowerModel 是否在真实数据上能成功跑通，并输出合理得分。

    Args:
        user_id: int，用户 ID
        user_histories: dict[user_id → List[item_id]]
        item_inputs: dict[item_id → input_dict]
        model: TwoTowerModel 实例
        device: 'cpu' or 'cuda'
    """

    if user_id not in user_histories:
        print(f"[Warning] user_id={user_id} not in user_histories.")
        return

    item_ids = user_histories[user_id]
    if len(item_ids) < 2 and popular_items:
        needed = 2 - len(item_ids)
        existing_set = set(item_ids)
        print(f"[Debug] existing item_ids for user {user_id}: {item_ids}")
        print(f"[Debug] popular_items (前10): {popular_items[:10]}")
        print(f"[Debug] items available in item_inputs: {len(item_inputs)}")

        # 检查过滤后的热门候选
        supplement = [iid for iid in popular_items if iid not in existing_set and iid in item_inputs]
        print(f"[Debug] 可用的热门补充项: {supplement[:10]}")

        item_ids += supplement[:needed]
        print(f"[Info] user {user_id} 补充了 {len(supplement[:needed])} 条热门 item → 当前历史数: {len(item_ids)}")

    if len(item_ids) < 2:
        print(f"[Error] user {user_id} 补充后仍不足 2 条 item")
        return

    # 正确构造历史和正样本（最后一条为正样本，其余为历史）
    history_ids = item_ids[:-1]
    pos_item_id = item_ids[-1]

    if pos_item_id not in item_inputs:
        print(f"[Error] Positive item {pos_item_id} not in item_inputs")
        return

    # 构造用户输入
    history_input_dicts = [
        item_inputs[iid] for iid in history_ids if iid in item_inputs
    ]
    if len(history_input_dicts) == 0:
        print(f"[Error] No valid history for user_id={user_id}")
        return

    # 构造负样本（从所有 item_inputs 中随机采一个该用户未看过的）
    candidate_ids = list(item_inputs.keys())
    neg_item_id = None
    attempts = 0
    while attempts < 100:
        sample = random.choice(candidate_ids)
        if sample not in item_ids:
            neg_item_id = sample
            break
        attempts += 1

    if neg_item_id is None:
        print("[Warning] Failed to sample a negative item.")
        return

    # 准备 item inputs
    pos_input = item_inputs[pos_item_id]
    neg_input = item_inputs[neg_item_id]

    # 调用模型 forward（注意是 batch）
    model.eval()
    with torch.no_grad():
        scores, _, _ = model(
            user_history_batch=[history_input_dicts, history_input_dicts],
            item_input_batch=[pos_input, neg_input]
        )

    print(f"\n[Result] user_id={user_id}")
    print(f"  Positive item_id: {pos_item_id}, Score: {scores[0].item():.4f}")
    print(f"  Negative item_id: {neg_item_id}, Score: {scores[1].item():.4f}")
    print("  → 正样本得分应高于负样本 ✅" if scores[0] > scores[1] else "  → ❌ 注意，模型可能未训练或异常")





df_user = pd.read_csv('../data/train.csv')
df_user = df_user.sort_values(by=['user_id', 'timestamp'])
user_histories = df_user.groupby('user_id')['item_id'].apply(list).to_dict()

user_id = list(user_histories.keys())[0]  # 动态选择第一个用户进行测试
df = pd.read_csv('../data/item_meta_processed.csv')

num_categories = int(df['main_category_encoded'].max()) + 1
num_stores = int(df['store_encoded'].max()) + 1
num_parent_asin = int(df['parent_asin_encoded'].max()) + 1

item_encoder = ItemEncoder(
    num_categories=num_categories,
    num_stores=num_stores,
    num_parent_asin=num_parent_asin,
    text_embedding_dim=384
)
user_encoder = UserEncoder(item_encoder)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

item_freq = Counter([iid for ids in user_histories.values() for iid in ids])
popular_items = [iid for iid, _ in item_freq.most_common(100)]  # top 100 热门 item

test_two_tower_model(
    user_id=user_id,
    user_histories=user_histories,
    item_inputs=torch.load("item_inputs.pt", map_location=device),
    model=TwoTowerModel(user_encoder, item_encoder).to(device),
    device=device,
    popular_items=popular_items
)
