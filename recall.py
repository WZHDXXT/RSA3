import pandas as pd

def compute_recall_at_k(submission_path, test_path, k=10):
    # Read prediction results and ground truth data
    submission = pd.read_csv(submission_path)
    test = pd.read_csv(test_path)

    # Build ground truth mapping from test: user_id -> set of true item_ids
    test_user_item = test.groupby('user_id')['item_id'].apply(set).to_dict()

    recalls = []

    for _, row in submission.iterrows():
        user = row['user_id']
        pred_items = list(map(int, row['item_id'].split(',')))[:k]

        true_items = test_user_item.get(user, set())
        if not true_items:
            continue  # Skip users with no ground truth

        hit_count = sum([1 for item in pred_items if item in true_items])
        recall = hit_count / len(true_items)
        recalls.append(recall)

    avg_recall = sum(recalls) / len(recalls) if recalls else 0.0
    print(f"Average Recall@{k}: {avg_recall:.4f}")
    return avg_recall

compute_recall_at_k("data/submission_test.csv", "test.csv", k=10)