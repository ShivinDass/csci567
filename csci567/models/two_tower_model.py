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


class PurchasesDataset(Dataset):
    '''dataset of purchases for contrastive learning'''
    def __init__(self, purchases_df, customers_index_dict, articles_index_dict, device=None):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.customer_ids = purchases_df["customer_id"].values
        self.purchases_ids = purchases_df["article_id"].values
        self.customers_index_dict = customers_index_dict
        self.articles_index_dict = articles_index_dict

    def __len__(self):
        return len(self.customer_ids)

    def __getitem__(self, idx):
        return torch.tensor(self.customers_index_dict[self.customer_ids[idx]]).to(self.device), \
            torch.tensor(self.articles_index_dict[np.random.choice(self.purchases_ids[idx])]).to(self.device)


class Encoder(nn.Module):
    '''Implements encoder to translate input embeddings into a meaningful representation'''
    def __init__(self, in_dim, out_dim, num_hidden_layers=0, hidden_layer_dim=12, activation=nn.LeakyReLU, device=None):
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
                 input_embed_dim=12, representation_embed_dim=12, device=None):
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
        self.loss_criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, contrastive_batch):
        queries, objects = contrastive_batch
        target_perm = torch.randperm(len(objects), requires_grad=False).to(self.device)
        # queries_latents = self.query_net(self.queries_embeddings(queries))
        # objects_latents = self.object_net(self.objects_embeddings(objects))
        queries_latents = self.queries_embeddings(queries)
        objects_latents = self.objects_embeddings(objects[target_perm])
        return torch.matmul(queries_latents, objects_latents.T), target_perm

    def predict(self, test_batch):
        with torch.no_grad():
            queries, objects = test_batch
            # queries_latents = self.query_net(self.queries_embeddings(queries))
            # objects_latents = self.object_net(self.objects_embeddings(objects))
            queries_latents = self.queries_embeddings(queries)
            objects_latents = self.objects_embeddings(objects)
            logits = torch.matmul(queries_latents, objects_latents.T)
        return torch.argmax(logits, dim=1)

    def loss(self, logits, targets):
        return self.loss_criterion(logits, targets)


class TwoTowerTrainer:
    '''training loop of two tower model'''
    def __init__(self, model, train_dataloader, test_dataloader, epochs, learning_rate, experiment_name="two_tower"):
        self.experiment_name = experiment_name
        self.save_model_path = os.path.join(
            os.environ["EXP_DIR"], f"{experiment_name}.pt")
        self.save_plot_path = os.path.join(
            os.environ["EXP_DIR"], f"{experiment_name}_plot.jpg")
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.epochs = epochs
        self.lr = learning_rate
        self.model = model.train()
        self.opt = torch.optim.Adam(self.model.parameters(),
                                    lr=learning_rate)
    
    def train(self):
        train_it = 0
        best_loss = float('inf')
        losses, val_accs = [], []
        for epoch in tqdm(range(self.epochs)):
            print(f"Run epoch {epoch}")
            for batch in tqdm(self.train_dataloader):
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
                if train_it % 500 == 0:
                    print(f"It {train_it}: Total Loss: {loss.cpu().detach()}")
                    val_acc = self.validate()
                    print(f"It {train_it}: Val Acc: {val_acc}")
                    val_accs.append(val_acc)
                train_it += 1
            # log the loss training curves
            plt.figure(figsize=(15, 5))
            ax1 = plt.subplot(121)
            ax1.plot(losses)
            ax1.title.set_text("loss")
            ax1 = plt.subplot(122)
            ax1.plot(val_accs)
            ax1.title.set_text("val acc")
            plt.savefig(self.save_plot_path)
        print("Done!")

    def validate(self):
        test_batch = None
        correct = 0
        for batch in self.test_dataloader:
            test_batch = batch
        prediction = self.model.predict(test_batch)
        for i in range(len(prediction)):
            correct += 1 if prediction[i] == i else 0
        return correct / len(prediction)



if __name__ == "__main__":
    # Init
    experiment_name = "two_tower_all_no_nn"
    print(f"Initializing experiment: {experiment_name}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 10000
    val_size = 100
    learning_rate = 2e-3

    # Load in training data
    print("Loading in training data")
    transactions_df = get_train_data(cutoff_date=None)
    purchases_df, customers_index_dict, articles_index_dict = get_customer_purchases(
        transactions_df, after="2020-08-23")
    training_purchases_df = purchases_df
    testing_purchases_df = purchases_df[:val_size]
    del transactions_df

    # Make dataset and dataloader
    print("Making purchases dataloader")
    training_purchases_dataset = PurchasesDataset(
        training_purchases_df, customers_index_dict, articles_index_dict, device=device)
    training_purchases_dataloader = DataLoader(
        training_purchases_dataset, batch_size=64, shuffle=True)
    testing_purchases_dataset = PurchasesDataset(
        testing_purchases_df, customers_index_dict, articles_index_dict, device=device)
    testing_purchases_dataloader = DataLoader(
        testing_purchases_dataset, batch_size=val_size, shuffle=True)

    # Make model and trainer
    print("Making model and trainer")
    two_tower_model = TwoTowerModel(
        len(customers_index_dict), len(articles_index_dict), device=device)
    count_parameters(two_tower_model)

    two_tower_trainer = TwoTowerTrainer(
        two_tower_model, training_purchases_dataloader, testing_purchases_dataloader, epochs, learning_rate, experiment_name)
    
    # Start training
    print("Starting training")
    two_tower_trainer.train()
