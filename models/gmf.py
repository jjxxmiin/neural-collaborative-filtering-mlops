import torch
import torch.nn as nn

class GMF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.num_users = cfg.dataset.num_users
        self.num_items = cfg.dataset.num_items
        self.latent_dim = cfg.dataset.latent_dim
        
        self.embedding_user = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        
        self.affine_output = nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = nn.Sigmoid()
        
    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_product = torch.mul(user_embedding, item_embedding)
        logits = self.affine_output(element_product)
        rating = self.logistic(logits)
        return rating
    
    
