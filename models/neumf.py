import torch

class NeuMF(torch.nn.Module):
    def __init__(self, cfg):
        super(NeuMF, self).__init__()
        
        self.num_users = cfg.dataset.num_users
        self.num_items = cfg.dataset.num_items
        self.latent_dim = cfg.dataset.latent_dim
        self.latent_dim_mf = cfg.dataset.latent_dim
        self.latent_dim_mlp = cfg.dataset.latent_dim

        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)

        self.fc_layers = torch.nn.ModuleList()
        for idx, (in_size, out_size) in enumerate(zip(cfg.model.neumf_layers[:-1], cfg.model.neumf_layers[1:])):
            self.fc_layers.append(torch.nn.Linear(in_size, out_size))

        self.affine_output = torch.nn.Linear(in_features=cfg.model.neumf_layers[-1] + cfg.model.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)

        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)  # the concat latent vector
        mf_vector =torch.mul(user_embedding_mf, item_embedding_mf)

        for idx, _ in enumerate(range(len(self.fc_layers))):
            mlp_vector = self.fc_layers[idx](mlp_vector)
            mlp_vector = torch.nn.ReLU()(mlp_vector)

        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def load_pretrain_weights(self, mlp, gmf):
        """Loading weights from trained MLP model & GMF model"""
        self.embedding_user_mlp.weight.data = mlp.embedding_user.weight.data
        self.embedding_item_mlp.weight.data = mlp.embedding_item.weight.data
        for idx in range(len(self.fc_layers)):
            self.fc_layers[idx].weight.data = mlp.fc_layers[idx].weight.data

        self.embedding_user_mf.weight.data = gmf.embedding_user.weight.data
        self.embedding_item_mf.weight.data = gmf.embedding_item.weight.data

        self.affine_output.weight.data = 0.5 * torch.cat([mlp.affine_output.weight.data, gmf.affine_output.weight.data], dim=-1)
        self.affine_output.bias.data = 0.5 * (mlp.affine_output.bias.data + gmf.affine_output.bias.data)
