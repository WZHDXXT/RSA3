import torch
from item_encoder import ItemEncoder  # 路径根据你项目结构调整
import pandas as pd

# 加载数据（假设这些列都已经是编码好的 int 值）
df = pd.read_csv('../data/item_meta_processed.csv')

# 直接用 max() + 1 作为 embedding 的输入词表大小（因为编码从 0 开始）
num_categories = int(df['main_category_encoded'].max()) + 1
num_stores = int(df['store_encoded'].max()) + 1
num_parent_asin = int(df['parent_asin_encoded'].max()) + 1

# 初始化模型
encoder = ItemEncoder(
    num_categories=num_categories,
    num_stores=num_stores,
    num_parent_asin=num_parent_asin,
    text_embedding_dim=384
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载 item 输入
item_inputs = torch.load("item_inputs.pt", map_location=device)

# 选择一个 item_id（比如取第一个，或指定的）
example_item_id = list(item_inputs.keys())[0]  # 或使用 example_item_id = 12345
print('ex', example_item_id)
inputs = item_inputs[example_item_id]
print(item_inputs[42864])
for k in inputs:
    inputs[k] = inputs[k].to(device)

# 假设你已经初始化好 encoder 模型并传入正确的 embedding size
with torch.no_grad():
    output_vec = encoder(
        category=inputs['category'],
        store=inputs['store'],
        parent_asin=inputs['parent_asin'],
        text_embedding=inputs['text_embedding'],
        num_vec=inputs['num_vec'],
        image_vec=inputs['image_vec']
    )

print(f"\n[Output] Embedding vector shape: {output_vec.shape}")
print(output_vec)  # 可选：查看具体值
