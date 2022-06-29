import torch
import random
import numpy as np
import pandas as pd
from copy import deepcopy
from torch.utils.data import Dataset, DataLoader


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


class MovieLensDataset():
    def __init__(self, ratings, cfg):
        super().__init__()
        
        self.cfg = cfg
        self.user_pool = set(ratings['userId'].unique())
        self.item_pool = set(ratings['itemId'].unique())
        
        preprocess_ratings = self._binarize(ratings)
        self.negatives = self._sample_negative(ratings)
        
        self.train_ratings, self.test_ratings = self._split_loo(preprocess_ratings)
        
    def get_train_loader(self):
        users, items, ratings = [], [], []
        train_ratings = pd.merge(self.train_ratings, self.negatives[['userId', 'negative_items']], on='userId')
        train_ratings['negatives'] = train_ratings['negative_items'].apply(lambda x: random.sample(x, self.cfg.dataset.num_negative))
        for row in train_ratings.itertuples():
            users.append(int(row.userId))
            items.append(int(row.itemId))
            ratings.append(float(row.rating))
            for i in range(self.cfg.dataset.num_negative):
                users.append(int(row.userId))
                items.append(int(row.negatives[i]))
                ratings.append(float(0))
        
        self.user_tensor = torch.LongTensor(users)
        self.item_tensor = torch.LongTensor(items)
        self.target_tensor = torch.FloatTensor(ratings)
        
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(users),
                                        item_tensor=torch.LongTensor(items),
                                        target_tensor=torch.FloatTensor(ratings))
        
        loader = DataLoader(dataset, 
                            batch_size=self.cfg.train.batch_size,
                            num_workers=self.cfg.train.num_workers,
                            shuffle=True)
    
        return loader
        
    def get_test_data(self):
        """create evaluate data"""
        test_ratings = pd.merge(self.test_ratings, self.negatives[['userId', 'negative_samples']], on='userId')
        test_users, test_items, negative_users, negative_items = [], [], [], []
        for row in test_ratings.itertuples():
            test_users.append(int(row.userId))
            test_items.append(int(row.itemId))
            for i in range(len(row.negative_samples)):
                negative_users.append(int(row.userId))
                negative_items.append(int(row.negative_samples[i]))
        return [torch.LongTensor(test_users), torch.LongTensor(test_items), torch.LongTensor(negative_users),
                torch.LongTensor(negative_items)]
        
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
    
    def _split_loo(self, ratings):
        """leave one out train/test split """
        ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
        test = ratings[ratings['rank_latest'] == 1]
        train = ratings[ratings['rank_latest'] > 1]
        assert train['userId'].nunique() == test['userId'].nunique()
        return train[['userId', 'itemId', 'rating']], test[['userId', 'itemId', 'rating']]

  
def get_data(cfg):
    ml1m_rating = pd.read_csv(cfg.dataset.ratings, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    
    train_dataset = MovieLensDataset(ml1m_rating, cfg=cfg)
    train_loader = train_dataset.get_train_loader()
    test_data = train_dataset.get_test_data()
    
    return train_loader, test_data