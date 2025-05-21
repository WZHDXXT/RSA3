import pandas as pd
import os
from collections import defaultdict

class MakeSequenceDataSet():
    """
    SequenceDataSet for custom implicit feedback data (from train.csv)

    Expects a CSV file with columns: user_id, item_id, timestamp
    Converts into user interaction sequences:
    - All but last interaction → train
    - Last interaction → validation
    """

    def __init__(self, data_path, filename="train.csv"):
        print('Reading train.csv...')
        self.df = pd.read_csv(os.path.join(data_path, filename))

        # Map user_id/item_id to continuous indices
        self.df['user_idx'] = pd.factorize(self.df['user_id'])[0]
        self.df['item_idx'] = pd.factorize(self.df['item_id'])[0]
        self.num_item = self.df['item_idx'].nunique()
        self.num_user = self.df['user_idx'].nunique()

        # Sort interactions by user and timestamp
        self.df = self.df.sort_values(['user_idx', 'timestamp'])

        # Build train/valid data
        self.user_train, self.user_valid = self.generate_sequence_data()

        print('Finish loading sequence data.')

    def generate_sequence_data(self):
        """
        Convert raw interactions into user-wise sequences.
        Each user: use all but last item for training, last for validation.
        """
        user_train = {}
        user_valid = {}

        grouped = self.df.groupby('user_idx')
        for user, group in grouped:
            seq = group['item_idx'].tolist()
            if len(seq) < 2:
                continue  # skip users with <2 interactions
            user_train[user] = seq[:-1]
            user_valid[user] = [seq[-1]]

        return user_train, user_valid

    def get_train_valid_data(self):
        """Return the processed train and valid dictionaries."""
        return self.user_train, self.user_valid, self.num_user, self.num_item