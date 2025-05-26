import pandas as pd
import json
from preprocess_all1 import load_and_process_item_meta
def process_categories_text(df, columns):
    """为 DataFrame 添加多个 categories_text 字段，支持通过 item_id 查找"""
    def parse(row, column):
        raw = row[column]
        if pd.isnull(raw) or raw in ['', '[]', [], {}]:
            return ''
        try:
            if isinstance(raw, str):
                raw = raw.replace("'", '"')
                parsed = json.loads(raw)
            else:
                parsed = raw

            if isinstance(parsed, dict):
                return " ".join([f"{k} {v}" for k, v in parsed.items() if v])
            elif isinstance(parsed, list):
                # 处理如 ['Material: 304 Steel', 'Unit: 5g']
                return " ".join([s.replace(":", "") for s in parsed if isinstance(s, str)])
            else:
                return ''
        except Exception:
            # fallback: 如果是普通字符串（无结构标记），直接返回清理后的文本
            if isinstance(raw, str):
                return raw.strip().replace(";", " ").replace(",", " ")
            return ''
    
    for col in columns:
        df[col+'_text'] = df.apply(lambda row: parse(row, col), axis=1)
    return df

# if __name__ == "__main__":
#     df = pd.read_csv("../data/item_meta.csv")
#     df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
#     df.set_index('item_id', inplace=True)  # Removed: do not set 'item_id' as index
    

#     column = 'title'
#     df = process_categories_text(df, column=column)
#     example_item_id = 1649
#     print("example_item_id:", example_item_id)
#     print(df.loc[example_item_id])
#     parent_asin = df.loc[example_item_id, column+'_text']
#     print(parent_asin)
#     # print("categories_text:", df[df['item_id'] == example_item_id]['details_text'].values[0])
# #{'Brand': 'HLIN', 'Capacity': '16 Ounces', 'Number of Items': '2', 'Product Care Instructions': 'Hand Wash Only', 'Reusability': 'Reusable', 'Package Dimensions': '9.45 x 8.54 x 3.82 inches; 4.23 Ounces', 'UPC': '722794955266'}

df = pd.read_csv("../data/item_meta.csv")
df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
df.set_index('item_id', inplace=True)
df = process_categories_text(df, columns=['description',
                                          'features',
                                          'title',
                                          'details'])
print(df.loc[326])
df2 = load_and_process_item_meta("../data/item_meta.csv").reset_index()

# 提取 df 中的 _text 字段
text_fields = ['description_text', 'features_text', 'title_text', 'details_text']
df_text = df[text_fields].reset_index()

# 合并并设置索引
df2 = df2.merge(df_text, on='item_id', how='left')
df2 = df2.set_index('item_id')

# 示例验证
print(df2.loc[326])

# 保存处理后的 DataFrame 为 CSV 文件
# df2.to_csv("../data/item_meta_processed.csv")
