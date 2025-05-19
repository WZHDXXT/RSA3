from threadpoolctl import threadpool_limits
threadpool_limits(1, "blas")

import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares
from collections import defaultdict
import numpy as np

# ========== 加载数据 ==========
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# 转成 category，构建 vocab
train_df['user_id'] = train_df['user_id'].astype("category")
train_df['item_id'] = train_df['item_id'].astype("category")

# 映射表
user_id_map = dict(enumerate(train_df['user_id'].cat.categories))
item_id_map = dict(enumerate(train_df['item_id'].cat.categories))

# 训练集编码
train_df['user_index'] = train_df['user_id'].cat.codes
train_df['item_index'] = train_df['item_id'].cat.codes

# 获取用户和物品数量
num_users = train_df['user_index'].nunique()
num_items = train_df['item_index'].nunique()

# 构造 item-user 矩阵 (implicit 推荐是 item × user)
item_user_matrix = sp.coo_matrix(
    (np.ones(len(train_df)), (train_df['item_index'], train_df['user_index'])),
    shape=(num_items, num_users)
).tocsr()

# 用户-物品矩阵（for recommend 时用）
user_item_matrix = item_user_matrix.T.tocsr()

# ========== 训练 ALS 模型 ==========
model = AlternatingLeastSquares(factors=64, regularization=0.01, iterations=15)
model.fit(item_user_matrix)

# ========== 准备测试集 ==========
# 映射到相同的 category
test_df['user_id'] = pd.Categorical(test_df['user_id'], categories=train_df['user_id'].cat.categories)
test_df['item_id'] = pd.Categorical(test_df['item_id'], categories=train_df['item_id'].cat.categories)

# 删除 cold-start
test_df = test_df.dropna(subset=['user_id', 'item_id'])

# 编码为索引
test_df['user_index'] = test_df['user_id'].cat.codes
test_df['item_index'] = test_df['item_id'].cat.codes

# 构建 ground truth
ground_truth = defaultdict(set)
for _, row in test_df.iterrows():
    u = int(row['user_index'])
    i = int(row['item_index'])
    ground_truth[u].add(i)

# ========== 评估 Recall@10 ==========
recalls = []
for u in ground_truth:
    # 推荐 top 10
    recommended = model.recommend(
        userid=u,
        user_items=user_item_matrix[u],  # ✅ 只取这个用户那一行
        N=10,
        filter_already_liked_items=True
    )

    recommended_items = set(i for i, _ in recommended)
    true_items = ground_truth[u]

    recall = len(recommended_items & true_items) / len(true_items)
    recalls.append(recall)

# 输出 Recall@10
print(f"✅ Recall@10: {np.mean(recalls):.4f}")


