import csv
import matplotlib.pyplot as plt
import numpy as np

def get_loss_vals(filename):
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
        train_losses, val_losses = rows[0], rows[1]
        val_losses = [float(val) for val in val_losses]
        train_losses = [float(val) for val in train_losses]
        return train_losses, val_losses

def plot_single_exp(network, idx, types=["small", "medium", "large"]):
    type_to_losses = {}
    
    fig, ax1 = plt.subplots()
    ax1.set_title(f'Losses for {network} on label {idx}')
    ax1.set_ylabel('Val Losses', alpha=1)
    ax1.tick_params(axis='y', labelcolor='C0')
    ax1.set_yscale('log')
    colors = ['C0', 'C1', 'C2']

    for i, ty in enumerate(types):
        filename = f"{network}_{ty}_l{idx}.csv"
        train_losses, val_losses = get_loss_vals(filename)
        type_to_losses[ty] = (train_losses, val_losses)

        # All experiments should have 10 epochs
        max_index = 70
        x_vals = [i for i in range(0, max_index)]
        train_losses, val_losses = train_losses[0:70], val_losses[0:70]
        
        #ax1.plot(x_vals, train_losses, colors[i], label=f"{ty}", alpha=0.5)
        ax1.plot(x_vals, val_losses, colors[i], alpha=1, label=f"{ty}")

    ax1.legend()
    plt.savefig(f"single_exp_{network}_{idx}.png")

def plot_across_train_valid(networks, idx):
    net_to_losses = {}
    # All experiments should have 10 epochs
    max_index = 70
    for i, network in enumerate(networks):
        fig, ax1 = plt.subplots()
        ax1.set_title(f'Train vs Valid of {network} on label {idx}')
        ax1.set_ylabel('Losses')
        ax1.tick_params(axis='y')
        ax1.set_yscale('log')
        colors = ['C0', 'C1', 'C2']
        
        min_loss, min_train_loss, min_loss_id = [10e10 for i in range(0,max_index)], None, None
        for ty in ["small", "medium", "large"]:
            filename = f"{network}_{ty}_l{idx}.csv"
            train_losses, val_losses = get_loss_vals(filename)
            if np.mean(val_losses[50:max_index]) < np.mean(min_loss[50:max_index]):
                min_loss = val_losses
                min_train_loss = train_losses
                min_loss_id = ty
        net_to_losses[network] = min_loss_id
        x_vals = [i for i in range(0, max_index)]
        ax1.plot(x_vals, min_train_loss[:max_index], colors[0], label=f"Train", alpha=1)
        ax1.plot(x_vals, min_loss[:max_index], colors[1], label=f"Validation", alpha=1)
        ax1.legend()
        plt.savefig(f"train_valid_{network}_{idx}.png")
        print(net_to_losses)

def plot_generalization_gap(networks, idx):
    fig, ax1 = plt.subplots()
    ax1.set_title(f'Generalization gap across networks on {idx}')
    ax1.set_ylabel('Validation Loss - Train Loss')
    ax1.tick_params(axis='y')
    
    ax1.set_yscale('log')
    colors = ['C0', 'C1', 'C2']

    net_to_losses = {}
    # All experiments should have 10 epochs
    max_index = 70
    for i, network in enumerate(networks):
        min_loss, min_train_loss, min_loss_id = [10e10 for i in range(0,max_index)], None, None
        for ty in ["small", "medium", "large"]:
            filename = f"{network}_{ty}_l{idx}.csv"
            train_losses, val_losses = get_loss_vals(filename)
            if np.mean(val_losses[50:max_index]) < np.mean(min_loss[50:max_index]):
                min_loss = np.array(val_losses)
                min_loss_id = ty
                min_train_loss = np.array(train_losses)

        net_to_losses[network] = min_loss_id
        x_vals = [i for i in range(0, max_index)]
        ax1.plot(x_vals, min_loss[:max_index]-min_train_loss[:max_index], colors[i], label=f"{network}", alpha=1)

    ax1.legend()
    plt.savefig(f"generalization_gap_{idx}.png")
    print(net_to_losses)


def plot_across_exp(networks, idx):
    fig, ax1 = plt.subplots()
    ax1.set_title(f'Best Performance across networks on label {idx}')
    ax1.set_ylabel('Validation Losses', color='C0')
    ax1.tick_params(axis='y', color='C0', labelcolor='C0')
    
    ax1.set_yscale('log')
    #ax2.set_yscale('log')
        
    colors = ['C0', 'C1', 'C2']

    net_to_losses = {}
    # All experiments should have 10 epochs
    max_index = 70
    for i, network in enumerate(networks):
        min_loss, min_loss_id = [10e10 for i in range(0,max_index)], None
        for ty in ["small", "medium", "large"]:
            filename = f"{network}_{ty}_l{idx}.csv"
            train_losses, val_losses = get_loss_vals(filename)
            if np.mean(val_losses[50:max_index]) < np.mean(min_loss[50:max_index]):
                min_loss = val_losses
                min_loss_id = ty
        net_to_losses[network] = min_loss_id

        x_vals = [i for i in range(0, max_index)]
        ax1.plot(x_vals, min_loss[:max_index], colors[i], label=f"{network}", alpha=1)

    ax1.legend()
    plt.savefig(f"across_exp_{idx}.png")
    print(net_to_losses)

for label in range(1,8):
    #for net in ["egnn", "nnConv", "GAT"]:
    #    plot_single_exp(net, label)
    #plot_across_train_valid(["nnConv", "egnn", "GAT"], label)
    #plot_across_exp(["nnConv", "egnn", "GAT"], label)
    networks = ["egnn", "nnConv", "GAT"] 
    plot_generalization_gap(networks, label)
