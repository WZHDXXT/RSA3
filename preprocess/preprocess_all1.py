import pandas as pd
import ast
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import json

def load_and_process_item_meta(path: str) -> pd.DataFrame:
    # Step 1: Load data
    df = pd.read_csv(path)
    
    # Step 2: Ensure unique item_id index
    df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
    df.set_index('item_id', inplace=True)

    # Step 3: Extract image_url from images field
    def extract_image_url(images_field):
        try:
            if pd.isnull(images_field):
                return None
            images = ast.literal_eval(images_field)  # 安全解析为 Python 对象
            if isinstance(images, list) and len(images) > 0:
                main_image = images[0]
                if isinstance(main_image, dict):
                    for key in ['hi_res', 'large', 'thumb']:
                        url = main_image.get(key)
                        if isinstance(url, str) and url.strip():
                            return url
        except Exception as e:
            print(f"Error: {e}")
        return None

    df['image_url'] = df['images'].apply(extract_image_url)

    def extract_video_title(videos_field):
        try:
            if pd.isnull(videos_field):
                return None
            videos = json.loads(videos_field.replace("'", '"'))
            if isinstance(videos, list) and len(videos) > 0:
                return videos[0].get('title')
        except:
            return None

    df['video_title'] = df['videos'].apply(extract_video_title)
    df['video_text'] = df['video_title'].apply(lambda x: f"Video: {x}" if pd.notnull(x) else "")

    # Step 3.1: Encode main_category as category
    le = LabelEncoder()
    df['main_category_encoded'] = le.fit_transform(df['main_category'].fillna("unknown"))
    # Step 3.2: Encode store as category
    le_store = LabelEncoder()
    df['store_encoded'] = le_store.fit_transform(df['store'].fillna("unknown"))
    # Step 3.3: Encode parent_asin as category
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