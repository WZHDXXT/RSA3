import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_process_item_meta(path: str) -> pd.DataFrame:
    # Step 1: 读取数据
    df = pd.read_csv(path)
    # Step 2: 构建 item_id 索引（假设唯一）
    df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
    df.set_index('item_id', inplace=True)

    # 处理 images 字段，提取 image_url
    import json
    def extract_image_url(images_field):
        try:
            if pd.isnull(images_field):
                return None
            images = json.loads(images_field.replace("'", '"'))
            if isinstance(images, list) and len(images) > 0:
                main_image = images[0]
                return main_image.get('hi_res') or main_image.get('large') or main_image.get('thumb')
        except:
            return None

    def extract_video_title(videos_field):
        try:
            if pd.isnull(videos_field):
                return None
            videos = json.loads(videos_field.replace("'", '"'))
            if isinstance(videos, list) and len(videos) > 0:
                return videos[0].get('title')
        except:
            return None

    df['image_url'] = df['images'].apply(extract_image_url)
    df['video_title'] = df['videos'].apply(extract_video_title)
    df['video_text'] = df['video_title'].apply(lambda x: f"Video: {x}" if pd.notnull(x) else "")

    # Step 3.1: main_category → 类别编码
    le = LabelEncoder()
    df['main_category_encoded'] = le.fit_transform(df['main_category'].fillna("unknown"))
    # Step 3.2: store → 类别编码
    le_store = LabelEncoder()
    df['store_encoded'] = le_store.fit_transform(df['store'].fillna("unknown"))
    # Step 3.3: parent_asin → 类别编码
    le_parent_asin = LabelEncoder()
    df['parent_asin_encoded'] = le_parent_asin.fit_transform(df['parent_asin'].fillna("unknown"))

    df['average_rating_encoded'] = pd.to_numeric(df['average_rating'], errors='coerce').fillna(0)
    df['rating_number_encoded'] = pd.to_numeric(df['rating_number'], errors='coerce').fillna(0)
    df['rating_number_log'] = (df['rating_number_encoded'] + 1).apply(np.log)
    df['price_encoded'] = pd.to_numeric(df['price'], errors='coerce').fillna(0)
    df['price_log'] = (df['price_encoded'] + 1).apply(np.log)
    return df

# df = load_and_process_item_meta("../data/item_meta.csv")
# print(df.loc[31420])