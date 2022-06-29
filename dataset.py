import torch
import random
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset


class UserItemRatingDataset(Dataset):
    """Wrapper, convert <user, item, rating> Tensor into Pytorch Dataset"""
    def __init__(self, user_tensor, item_tensor, target_tensor):
        """
        args:
            target_tensor: torch.Tensor, the corresponding rating for <user, item> pair
        """
        self.user_tensor = user_tensor
        self.item_tensor = item_tensor
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)


class MovieLensDataset(Dataset):
    def __init__(self, ratings, num_negatives, train=True):
        super().__init__()
        
        self.user_pool = set(ratings['userId'].unique())
        self.item_pool = set(ratings['itemId'].unique())
        
        preprocess_ratings = self._binarize(ratings)
        negatives = self._sample_negative(ratings)
        
        if train:
            split_ratings = self._split_train(preprocess_ratings)
        else:
            split_ratings = self._split_test(preprocess_ratings)
        
        users, items, ratings = [], [], []
        split_ratings = pd.merge(split_ratings, negatives[['userId', 'negative_items']], on='userId')
        split_ratings['negatives'] = split_ratings['negative_items'].apply(lambda x: random.sample(x, num_negatives))
        for row in split_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(num_negatives):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
        
        self.user_tensor = torch.LongTensor(users)
        self.item_tensor = torch.LongTensor(items)
        self.target_tensor = torch.FloatTensor(ratings)
        
    def _binarize(self, ratings):
        """binarize into 0 or 1, imlicit feedback"""
        ratings = deepcopy(ratings)
        ratings['rating'][ratings['rating'] > 0] = 1.0
        return ratings
    
    def _sample_negative(self, ratings):
        """return all negative items & 100 sampled negative items"""
        interact_status = ratings.groupby('userId')['itemId'].apply(set).reset_index().rename(
            columns={'itemId': 'interacted_items'})
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: self.item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, 99))
        return interact_status[['userId', 'negative_items', 'negative_samples']]
    
    def _split_train(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        train = ratings[ratings['rank_latest'] > 1]
        
        return train[['userId', 'itemId', 'rating']]
    
    def _split_test(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        
        return test[['userId', 'itemId', 'rating']]
    
    def __getitem__(self, index):
        return self.user_tensor[index], self.item_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.user_tensor.size(0)