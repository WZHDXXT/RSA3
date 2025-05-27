import torch
from item_encoder import ItemEncoder  # Adjust the path according to your project structure
import pandas as pd

# Load the data (assuming the columns are already encoded as int values)
df = pd.read_csv('../data/item_meta_processed.csv')

# Use max() + 1 as the embedding input vocabulary size (since encoding starts from 0)
num_categories = int(df['main_category_encoded'].max()) + 1
num_stores = int(df['store_encoded'].max()) + 1
num_parent_asin = int(df['parent_asin_encoded'].max()) + 1

# Initialize the model
encoder = ItemEncoder(
    num_categories=num_categories,
    num_stores=num_stores,
    num_parent_asin=num_parent_asin,
    text_embedding_dim=384
)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load item inputs
item_inputs = torch.load("item_inputs.pt", map_location=device)

# Select an item_id (e.g., use the first one or a specified one)
example_item_id = list(item_inputs.keys())[0]  # or use example_item_id = 12345
print('ex', example_item_id)
inputs = item_inputs[example_item_id]
print(item_inputs[42864])
for k in inputs:
    inputs[k] = inputs[k].to(device)

# Assuming the encoder model is already initialized with the correct embedding size
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
print(output_vec)  # Optional: view actual values
