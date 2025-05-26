import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder

def load_and_process_item_meta(path: str) -> pd.DataFrame:
    # Step 1: Load data
    df = pd.read_csv(path)
    # Step 2: Build item_id index (assumed unique)
    df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
    df.set_index('item_id', inplace=True)

    # Step 3.1: Encode main_category as category
    le = LabelEncoder()
    df['main_category_encoded'] = le.fit_transform(df['main_category'].fillna("unknown"))
    # Step 3.2: Encode store as category
    le_store = LabelEncoder()
    df['store_encoded'] = le_store.fit_transform(df['store'].fillna("unknown"))
    # Step 3.3: Encode parent_asin as category
    le_parent_asin = LabelEncoder()
    df['parent_asin_encoded'] = le_parent_asin.fit_transform(df['parent_asin'].fillna("unknown"))

    # ********
    embedding_dim = 16
    # Step 4.1: Build embedding lookup for main_category (16-dim)
    num_categories = df['main_category_encoded'].nunique()
    # Step 4.2: Build store embedding lookup (16-dim)
    num_stores = df['store_encoded'].nunique()
    # Step 4.3: Build parent_asin embedding lookup (16-dim)
    num_parent_asins = df['parent_asin_encoded'].nunique()

    store_embedding = nn.Embedding(num_stores, embedding_dim)
    main_category_embedding = nn.Embedding(num_categories, embedding_dim)
    parent_asin_embedding = nn.Embedding(num_parent_asins, embedding_dim)

    return df

df = load_and_process_item_meta("../data/item_meta.csv")
print(df.loc[31420])