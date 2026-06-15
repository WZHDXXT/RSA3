#  Two-Tower Multi-Modal Recommendation System

This repository implements a modular two-tower recommendation framework designed for multi-modal item metadata, including textual, visual, numerical, and categorical information.

##  Project Structure

```
two_tower/
├── data/                    # Datasets and preprocessed features
│   ├── train.csv            # User-item interaction logs
│   ├── test.csv             # Users for prediction
│   ├── submission.csv       # Output prediction format
│   ├── item_meta.csv        # Raw item metadata
│   ├── item_meta_processed.csv  # Processed item metadata
│   └── item_inputs.pt       # Precomputed item vectors (skip vectorize)
│
├── models/                  # Model definitions
│   ├── item_encoder.py      # Item tower with fusion strategies
│   ├── user_encoder.py      # User tower with pooling strategies
│   └── two_tower_model.py   # Combined model
│
├── preprocess/              # Data preprocessing scripts
│   ├── vectorize.py         # Embedding and feature generation (optional)
│   ├── process_all2.py      # Metadata parsing to generate item_meta_processed.csv
│   ├── preprocess_*.py      # Specific steps for multi-modal input
│
├── utils/                   
│   ├── config.py            # Configuration and hyperparameters
│   ├── data_loader.py       # Dataset loading and collate functions
│
├── losses.py                # BCE, BPR, Hinge loss implementations
├── train.py                 # Main training script
├── evaluate.py              # Generate top-K recommendations
├── recall.py                # Recall@10 evaluation
├── infer.py                 # Submission generation
├── main.py                  # Whole procedure 
```


##  Running the Project

1. **(Optional) Preprocess Metadata**
   > Already done: `data/item_inputs.pt` is precomputed.

2. **Train Model**
   ```bash
   python train.py
   ```

3. **Evaluate Recall@10**
   ```bash
   python evaluate.py
   python recall.py
   ```

4. **Generate Submission**
   ```bash
   python infer.py
   ```
5. **Whole procedure**
   ```bash
   python main.py
   ```
##  Evaluation Metric

We use **Recall@10** on the test set to evaluate top-K recommendation quality.