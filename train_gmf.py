import os
import hydra
from hydra import utils
import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
from mlflow import pytorch as ml_torch
from tqdm import tqdm
from models import GMF
from dataset import get_data
from utils import Evaluator, log_params_from_omegaconf_dict
from omegaconf import OmegaConf

os.environ['HYDRA_FULL_ERROR'] = '1'

@hydra.main(version_base=None, config_path="conf", config_name="gmf")
def main(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    train_loader, test_data = get_data(cfg)
    train_steps_per_epoch = len(train_loader)
    
    evaluator = Evaluator(top_k=10)
    
    gmf = GMF(cfg).to(cfg.train.device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(gmf.parameters(), 
                           lr=cfg.train.learning_rate)
    
    mlflow.set_tracking_uri('file://' + utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.mlflow.runname)
    with mlflow.start_run():
        min_loss = 1000
        
        for e in range(cfg.train.epoch):
            log_params_from_omegaconf_dict(cfg)

            gmf.train()
            
            train_loss = 0
            train_pbar = tqdm(enumerate(train_loader), total=train_steps_per_epoch)
            
            for i, (user, item, rating) in train_pbar:
                user = user.to(cfg.train.device)
                item = item.to(cfg.train.device)
                rating = rating.to(cfg.train.device)

                optimizer.zero_grad()
                
                rating_pred = gmf(user, item)
                
                loss = criterion(rating_pred.view(-1), rating)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_description(f"Epoch: [{e}/{cfg.train.epoch}] | Train Loss: {train_loss / (i + 1)} | ")
        
            step = (e + 1) * train_steps_per_epoch
            train_loss /= train_steps_per_epoch 
            mlflow.log_metric('train_loss', train_loss, step=step)
            
            hit_ratio, ndcg = evaluator(gmf, test_data)
            mlflow.log_metric('hit_ratio', hit_ratio, step=step)
            mlflow.log_metric('ndcg', ndcg, step=step)
            
            if train_loss < min_loss:
                min_loss = train_loss
                torch.save(gmf.state_dict(), cfg.train.best_checkpoint)
                ml_torch.log_model(mlp, 'ml_model')
                
            torch.save(gmf.state_dict(), cfg.train.latest_checkpoint)
            
if __name__ == "__main__":
    main()