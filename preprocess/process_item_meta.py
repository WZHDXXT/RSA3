import pandas as pd
import json
from sentence_transformers import SentenceTransformer
import numpy as np

def process_item_meta(input_path, output_path):
    df = pd.read_csv(input_path)

    # Process categorical features
    for col in ['main_category', 'store', 'parent_asin']:
        df[col] = df[col].astype('category').cat.codes
        df[col] = df[col].apply(lambda x: x if x >= 0 else 0)

    # Process numerical features, fill missing values
    df['average_rating'] = df['average_rating'].fillna(0)
    df['rating_number'] = df['rating_number'].fillna(0)
    df['price'] = df['price'].fillna(0)

    # Extract text features
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

    df['full_text'] = df.apply(concat_text, axis=1)
    df['text_embedding'] = model.encode(
        df['full_text'].tolist(),
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    ).tolist()

    # Save processed data
    df.to_pickle(output_path)

if __name__ == "__main__":
    process_item_meta('data/item_meta.csv', 'data/processed_item_meta.pkl')