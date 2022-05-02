from csci567.utils.data_utils import *

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from prettytable import PrettyTable
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def get_customer_purchases(transactions_df, after="2020-08-23"):
    ''' returns purchases_df: customer_id -> list of purchased article_ids'''
    recent_purchases = transactions_df.loc[transactions_df.t_dat >= pd.to_datetime(after)]
    customers_index_dict = {cust_id: id for id, cust_id in enumerate(
        recent_purchases['customer_id'].unique())}
    articles_index_dict = {art_id: id for id, art_id in enumerate(
        recent_purchases['article_id'].unique())}
    return recent_purchases.groupby("customer_id").article_id.apply(list).reset_index(), \
        customers_index_dict, \
        articles_index_dict

def predict(queries, objects, two_tower_model):
    '''return csv of queries and list of predicted objects'''
    # Run all queries and objects through two_tower_model
    # Take argmax of logits
    pass


class PurchasesDataset(Dataset):
    '''dataset of purchases for contrastive learning'''
    def __init__(self, purchases_df, customers_index_dict, articles_index_dict):
        self.customer_ids = purchases_df["customer_id"].values
        self.purchases_ids = purchases_df["article_id"].values
        self.customers_index_dict = customers_index_dict
        self.articles_index_dict = articles_index_dict

    def __len__(self):
        return len(self.customer_ids)

    def __getitem__(self, idx):
        return self.customers_index_dict[self.customer_ids[idx]], \
            self.articles_index_dict[np.random.choice(self.purchases_ids[idx])]


class Encoder(nn.Module):
    '''Implements encoder to translate input embeddings into a meaningful representation'''
    def __init__(self, in_dim, out_dim, num_hidden_layers=3, hidden_layer_dim=32, activation=nn.LeakyReLU, device=None):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        linear_layers = nn.ModuleList()
        linear_layers.append(nn.Linear(in_dim, hidden_layer_dim))
        linear_layers.append(activation())
        for _ in range(num_hidden_layers):
            linear_layers.append(nn.Linear(hidden_layer_dim, hidden_layer_dim))
            linear_layers.append(activation())
        linear_layers.append(nn.Linear(hidden_layer_dim, out_dim))
        self.model = nn.Sequential(*linear_layers).to(self.device)

    def forward(self, x):
        return self.model(x)


class TwoTowerModel(nn.Module):
    '''Implements two tower contrastive learning'''
    def __init__(self, num_queries, num_objects,
                 input_embed_dim=40, representation_embed_dim=32, device=None):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.queries_embeddings = nn.Embedding(
            num_queries, input_embed_dim).to(self.device)
        self.objects_embeddings = nn.Embedding(
            num_objects, input_embed_dim).to(self.device)
        self.query_net = Encoder(
            input_embed_dim, representation_embed_dim, device=device).to(self.device)
        self.object_net = Encoder(
            input_embed_dim, representation_embed_dim, device=device).to(self.device)
        self.loss_criterion = nn.CrossEntropyLoss()

    def forward(self, contrastive_batch):
        queries, objects = contrastive_batch
        queries_latents = self.query_net(self.queries_embeddings(queries))
        objects_latents = self.object_net(self.objects_embeddings(objects))
        target_perm = torch.randperm(len(objects_latents))
        objects_latents = objects_latents[target_perm]
        return torch.matmul(queries_latents, objects_latents.T), target_perm

    def loss(self, logits, targets):
        return self.loss_criterion(logits, targets)


class TwoTowerTrainer:
    '''training loop of two tower model'''
    def __init__(self, model, dataloader, epochs, learning_rate, experiment_name="two_tower"):
        self.experiment_name = experiment_name
        self.save_model_path = os.path.join(
            os.environ["EXP_DIR"], f"{experiment_name}.pt")
        self.save_plot_path = os.path.join(
            os.environ["EXP_DIR"], f"{experiment_name}_plot.jpg")
        self.dataloader = dataloader
        self.epochs = epochs
        self.lr = learning_rate
        self.model = model.train()
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=learning_rate)
    
    def train(self):
        train_it = 0
        best_loss = float('inf')
        losses = []
        for epoch in tqdm(range(self.epochs)):
            print(f"Run epoch {epoch}")
            for batch in tqdm(self.dataloader):
                self.opt.zero_grad()
                logits, targets = self.model(batch)
                loss = self.model.loss(logits, targets)
                loss.backward()
                self.opt.step()
                losses.append(loss.cpu().detach())
                new_loss = sum(losses[-10:]) / len(losses[-10:])
                if new_loss < best_loss:
                    print(f"saving checkpoint. new best loss: {new_loss}")
                    torch.save(self.model, self.save_model_path)
                    best_loss = new_loss
                if train_it % 1000 == 0:
                    print(f"It {train_it}: Total Loss: {loss.cpu().detach()}")
                train_it += 1
            # log the loss training curves
            plt.figure(figsize=(15, 5))
            ax1 = plt.subplot(111)
            ax1.plot(losses)
            ax1.title.set_text("loss")
            plt.savefig(self.save_plot_path)
        print("Done!")


if __name__ == "__main__":
    # Init
    experiment_name = "two_tower"
    print(f"Initializing experiment: {experiment_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10
    learning_rate = 1e-4

    # Load in training data
    print("Loading in training data")
    transactions_df = get_train_data(cutoff_date=None)
    purchases_df, customers_index_dict, articles_index_dict = get_customer_purchases(
        transactions_df, after="2020-08-23")

    # Make dataset and dataloader
    print("Making purchases dataloader")
    purchases_dataset = PurchasesDataset(
        purchases_df, customers_index_dict, articles_index_dict)
    purchases_dataloader = DataLoader(
        purchases_dataset, batch_size=32, shuffle=True)

    # Make model and trainer
    print("Making model and trainer")
    two_tower_model = TwoTowerModel(
        len(customers_index_dict), len(articles_index_dict), device=device)
    count_parameters(two_tower_model)

    two_tower_trainer = TwoTowerTrainer(
        two_tower_model, purchases_dataloader, epochs, learning_rate, experiment_name)
    
    # Start training
    print("Starting training")
    two_tower_trainer.train()
