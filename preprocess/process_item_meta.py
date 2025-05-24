import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import numpy as np

def process_item_meta(input_path, output_path):
    df = pd.read_csv(input_path)

    # 处理类别型特征
    df['main_category'] = df['main_category'].astype('category').cat.codes
    df['store'] = df['store'].astype('category').cat.codes
    df['parent_asin'] = df['parent_asin'].astype('category').cat.codes

    # 处理数值型特征，填充缺失值
    df['average_rating'] = df['average_rating'].fillna(0)
    df['rating_number'] = df['rating_number'].fillna(0)
    df['price'] = df['price'].fillna(0)

    # 提取文本特征
    model = SentenceTransformer('all-MiniLM-L6-v2')

    def concat_text(row):
        title = row['title'] if pd.notnull(row['title']) else ''
        description = row['description'] if pd.notnull(row['description']) else ''

        try:
            features = json.loads(row['features']) if pd.notnull(row['features']) else []
            features_str = ' '.join(features) if isinstance(features, list) else str(features)
        except:
            features_str = ''

        try:
            details = row['details']
            if isinstance(details, dict):
                details_str = ' '.join([f"{k} {v}" for k, v in details.items()])
            elif isinstance(details, str):
                d = json.loads(details.replace("'", '"'))
                details_str = ' '.join([f"{k} {v}" for k, v in d.items()])
            else:
                details_str = ''
        except:
            details_str = ''

        return f"{title} {description} {features_str} {details_str}"

    df['text_embedding'] = df.apply(concat_text, axis=1).apply(lambda x: model.encode(x))

    # 保存处理后的数据
    df.to_pickle(output_path)

if __name__ == "__main__":
    process_item_meta('../data/item_meta.csv', '../data/processed_item_meta.pkl')