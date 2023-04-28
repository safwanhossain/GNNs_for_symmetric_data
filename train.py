import torch
import torch.nn as nn
import dgl 
import csv
import numpy as np
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch.utils.data import random_split
from torch_geometric.datasets import QM9

from models import GATNetwork, EGNNetwork, NNConvNetwork

if torch.cuda.is_available:
    on_gpu = True
    print("Using gpu resources")

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params

def train_model_qm9(model, train_loader, valid_loader, epochs, idx, batch_size):
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    if on_gpu:
        model.cuda()

    train_losses, val_losses = [], []
    for epoch in range(epochs):
        avg_loss = 0
        total_graphs = 0
        model.train()
        for i, data in enumerate(train_loader):
            if on_gpu:
                data = data.cuda()
            pred = model(data)

            optimizer.zero_grad()
            loss_train = loss_fn(pred, data.y[:, idx].unsqueeze(1))
            loss_train.backward()
            optimizer.step()
            avg_loss += loss_train.item()
            total_graphs += data.num_graphs
        
            if i != 0 and i % 200 == 0:
                avg_loss /= total_graphs
                
                val_loss = 0
                total_val_graphs = 0
                model.eval()
                for valid_data in valid_loader:
                    if on_gpu:
                        valid_data = valid_data.cuda()
                    pred = model(valid_data)
                    loss_valid = loss_fn(pred, valid_data.y[:, idx].unsqueeze(1))
                    val_loss += loss_valid.item()*batch_size
                    total_val_graphs += valid_data.num_graphs
                val_loss /= total_val_graphs
    
                train_losses.append(avg_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch} iter {i}: train_loss: {avg_loss}, validation_loss: {val_loss}")
                avg_loss, total_graphs = 0, 0
                model = model.train()

    return train_losses, val_losses

if __name__ == "__main__":
    qm9_dset = QM9('./')
    batch_size = 64
    train_set, valid_set = random_split(qm9_dset,[100000, 30831])
    train_loader = geom_data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = geom_data.DataLoader(valid_set, batch_size=batch_size, shuffle=True)
    #model, name = NNConvNetwork(14, 4), "nnConv"
    model, name = EGNNetwork(11, 4), "egnn"
    #model, name = GATNetwork(14, 4), "GAT"
    print(f"Model {name} has {get_num_params(model)} parameters")
    for idx in range(1,10):
        print(f"Running id {idx}")
        train_losses, val_losses = train_model_qm9(model, train_loader, valid_loader, 10, idx, batch_size)
        with open(name+f"_small_l{idx}.csv", 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile, delimiter=',')
            csv_writer.writerow(train_losses)
            csv_writer.writerow(val_losses)
    
