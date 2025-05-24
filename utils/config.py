# utils/config.py

import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Two-Tower Recommender Config")

    # 数据路径
    parser.add_argument('--train_path', type=str, default='data/train.csv')
    parser.add_argument('--test_path', type=str, default='data/test.csv')
    parser.add_argument('--item_meta_pkl', type=str, default='data/item_meta.pkl')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_seq_len', type=int, default=50)

    # 嵌入维度
    parser.add_argument('--embedding_dim', type=int, default=64)

    # 模型保存路径
    parser.add_argument('--save_path', type=str, default='checkpoints/two_tower.pt')

    # 推理用参数
    parser.add_argument('--top_k', type=int, default=10)

    args = parser.parse_args()
    return args
