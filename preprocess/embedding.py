import torch
import pandas as pd
from tqdm import tqdm
import pickle  # Used to save data
from sentence_transformers import SentenceTransformer

import torch
from numpy import log1p
# from utils.image import load_and_preprocess_image
from PIL import Image
import requests
from torchvision import transforms
from transformers import CLIPModel

def get_image_encoder(device='cpu'):
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    model = model.vision_model.eval().to(device)
    return model

def load_and_preprocess_image(image_url, image_size=224):
    """
    # Load image from URL and preprocess it for CLIP-compatible input
    # Returns: Tensor [3, H, W] (float32, normalized)
    """
    try:
        response = requests.get(image_url, timeout=5)
        image = Image.open(response.raw).convert('RGB')
    except Exception as e:
        raise RuntimeError(f"Can't load: {image_url}，: {e}")

    preprocess = transforms.Compose([
        transforms.Resize(image_size, interpolation=Image.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # [0,1]
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])
    return preprocess(image)  # Tensor shape: [3, image_size, image_size]


def build_item_embedding_input(row, sentence_bert_model, image_encoder=None, device='cpu'):
    # Categorical features
    category = torch.tensor([row['main_category_encoded']], dtype=torch.long).to(device)
    store = torch.tensor([row['store_encoded']], dtype=torch.long).to(device)
    parent_asin = torch.tensor([row['parent_asin_encoded']], dtype=torch.long).to(device)

    # Text concatenation
    full_text = " ".join([
        str(row.get('title_text', '')),
        str(row.get('description_text', '')),
        str(row.get('features_text', '')),
        str(row.get('categories_text', '')),
        str(row.get('video_text', '')),
        str(row.get('details_text', ''))
    ]).strip()
    text_vec = sentence_bert_model.encode(full_text, convert_to_tensor=True).unsqueeze(0).to(device)  # [1, 384]

    # Numerical features
    price_log = log1p(row.get('price_encoded', 0.0))
    rating_log = log1p(row.get('rating_number_encoded', 0.0))
    avg_rating = row.get('average_rating_encoded', 0.0)
    num_vec = torch.tensor([[rating_log, price_log, avg_rating]], dtype=torch.float32).to(device)  # [1, 3]

    # Image features
    image_url = row.get('image_url', None)
    if image_encoder is not None and image_url:
        try:
            image_tensor = load_and_preprocess_image(image_url)  # shape: [3, H, W]
            with torch.no_grad():
                image_vec = image_encoder(image_tensor.unsqueeze(0).to(device))  # [1, 512]
        except Exception:
            image_vec = torch.zeros((1, 512), dtype=torch.float32).to(device)
    else:
        image_vec = torch.zeros((1, 512), dtype=torch.float32).to(device)

    return {
        'category': category,
        'store': store,
        'parent_asin': parent_asin,
        'text_embedding': text_vec,
        'num_vec': num_vec,
        'image_vec': image_vec
    }


def main():
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Using device: {device}")

    # Load models
    print("[Info] Loading models...")
    sentence_bert_model = SentenceTransformer('sentence-transformers/paraphrase-MiniLM-L3-v2', device=device)
    image_encoder = get_image_encoder(device)

    # Load data
    print("[Info] Loading data...")
    df = pd.read_csv('../data/item_meta_processed.csv')
    df = df.drop_duplicates(subset=['item_id']).reset_index(drop=True)
    df.set_index('item_id', inplace=True)

    item_inputs = {}

    print("[Info] Building embedding inputs...")
    for i, (item_id, row) in enumerate(df.iterrows()):
        print(f"[Debug] Processing item_id={item_id}\n")
        if i > 3:
            break
        try:
            inputs = build_item_embedding_input(
                row=row,
                sentence_bert_model=sentence_bert_model,
                image_encoder=image_encoder,
                device=device
            )
            item_inputs[item_id] = inputs
        except Exception as e:
            print(f"[Warning] item_id={item_id} failed: {e}")

    print(f"[Info] Finished. Successfully processed {len(item_inputs)} items.")

    # Debug: print first few item_inputs
    print("\n[Debug] Sample item_inputs:")
    for i, (item_id, inputs) in enumerate(item_inputs.items()):
        print(f"\nItem ID: {item_id}")
        for k, v in inputs.items():
            print(f"  {k}: shape {tuple(v.shape)}")
        if i >= 2:
            break

    # Save as PyTorch file
    torch.save(item_inputs, "item_inputs.pt")

    # # Optional: Save as pickle
    # with open("item_inputs.pkl", "wb") as f:
    #     pickle.dump(item_inputs, f)

    # print("[Info] Embedding inputs saved.")

if __name__ == "__main__":
    main()
