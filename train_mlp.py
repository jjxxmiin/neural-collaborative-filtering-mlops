import os
import hydra
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.gmf import GMF
from dataset import MovieLensDataset
from omegaconf import DictConfig, OmegaConf, ListConfig

os.environ['HYDRA_FULL_ERROR'] = '1'

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)

@hydra.main(version_base=None, config_path="conf", config_name="gmf")
def main(cfg : DictConfig):
    print(OmegaConf.to_yaml(cfg))

    ml1m_rating = pd.read_csv(cfg.dataset.ratings, sep='::', header=None, names=['uid', 'mid', 'rating', 'timestamp'],  engine='python')

    user_id = ml1m_rating[['uid']].drop_duplicates().reindex()
    user_id['userId'] = np.arange(len(user_id))
    ml1m_rating = pd.merge(ml1m_rating, user_id, on=['uid'], how='left')
    item_id = ml1m_rating[['mid']].drop_duplicates()
    item_id['itemId'] = np.arange(len(item_id))
    ml1m_rating = pd.merge(ml1m_rating, item_id, on=['mid'], how='left')
    ml1m_rating = ml1m_rating[['userId', 'itemId', 'rating', 'timestamp']]
    
    train_dataset = MovieLensDataset(ml1m_rating, 
                                     num_negatives=cfg.dataset.num_negative)
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=cfg.train.batch_size,
                              num_workers=cfg.train.num_workers,
                              shuffle=True)
    
    model = GMF(cfg).to(cfg.train.device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), 
                           lr=cfg.train.learning_rate)
    
    train_steps_per_epoch = len(train_loader)
    
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        min_loss = 1000
        
        for e in range(cfg.train.epoch):
            log_params_from_omegaconf_dict(cfg)

            train_loss = 0
            train_pbar = tqdm(enumerate(train_loader), total=train_steps_per_epoch)
            
            for i, (user, item, rating) in train_pbar:
                user = user.to(cfg.train.device)
                item = item.to(cfg.train.device)
                rating = rating.to(cfg.train.device)

                optimizer.zero_grad()
                
                rating_pred = model(user, item)
                
                loss = criterion(rating_pred.view(-1), rating)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_description(f"Epoch: [{e}/{cfg.train.epoch}] | Train Loss: {train_loss / (i + 1)} | ")
        
            step = (e + 1) * train_steps_per_epoch
            train_loss /= train_steps_per_epoch 
            mlflow.log_metric('train_loss', train_loss, step=step)
            
            if train_loss < min_loss:
                min_loss = train_loss
                torch.save(model.state_dict(), cfg.train.best_checkpoint)
                
            torch.save(model.state_dict(), cfg.train.latest_checkpoint)
            
if __name__ == "__main__":
    main()