import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_process_item_meta(path: str) -> pd.DataFrame:
    # Step 1: Load data
    df = pd.read_csv(path)
    # Step 2: Build item_id index (assumed unique)
    df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
    df.set_index('item_id', inplace=True)

    # Process images field to extract image_url
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
    return df

df = load_and_process_item_meta("../data/item_meta.csv")
print(df.loc[42955])