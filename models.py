import torch
import torch.nn as nn
import dgl 
import torch_geometric
import torch_geometric.nn as geom_nn
import torch_geometric.data as geom_data
from torch_geometric.nn import NNConv, GATConv, global_add_pool

class GATNetwork(nn.Module):
    def __init__(self, node_f, edge_f, conv_layers=3, **kwargs):
        super().__init__()
        self.out_size = 32//2
        self.pos = 3
        self.conv_layers = conv_layers
        heads = 2
        #heads = 4
        self.hidden_nodes = [64//2, 128//2, 64//2]
        self.hidden_edges = [64//2, 128//2, 64//2]

        self.node_layer = nn.Sequential(
            nn.Linear(node_f, self.out_size),
            nn.ReLU()
        )
        self.edge_layer = nn.Sequential(
            nn.Linear(edge_f, self.out_size),
            nn.ReLU()
        )
        
        self.gat_layers = nn.ModuleList([\
            dgl.nn.pytorch.conv.EGATConv(
                in_node_feats=self.out_size, 
                in_edge_feats=self.out_size,
                out_node_feats=self.hidden_nodes[0],
                out_edge_feats=self.hidden_edges[0],
                num_heads=heads
            )
        ])
        
        for l in range(conv_layers-1):
            layer = dgl.nn.pytorch.conv.EGATConv(
                    in_node_feats=self.hidden_nodes[l]*heads, 
                    in_edge_feats=self.hidden_edges[l]*heads,
                    out_node_feats=self.hidden_nodes[l+1],
                    out_edge_feats=self.hidden_edges[l+1],
                    num_heads=heads
            )
            self.gat_layers.append(layer)

        self.linear_node = nn.Sequential(
            nn.Linear((self.hidden_nodes[-1])*heads, self.hidden_nodes[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_nodes[-1], 1)
        )

    def forward(self, data):
        batch, feat, edge_index, coord, edges = (\
            data.batch, data.x, data.edge_index, data.pos, data.edge_attr
        )
        g = dgl.graph((edge_index[0], edge_index[1]))
        feat = torch.concat([feat, coord], dim=1)
        num_feats, num_edges = feat.shape[0], edges.shape[0]
        
        feat = self.node_layer(feat)
        edges = self.edge_layer(edges)
        feat = feat.reshape(num_feats, -1)
        edges = edges.reshape(num_edges, -1)

        for l in range(self.conv_layers):
            feat, edges = self.gat_layers[l](g, feat, edges)
            feat = torch.nn.functional.relu(feat)
            edges = torch.nn.functional.relu(edges)
            feat = feat.reshape(num_feats, -1)
            edges = edges.reshape(num_edges, -1)
            #feat = torch.nn.functional.dropout(feat, p=0.2)
            #edges = torch.nn.functional.dropout(edges, p=0.2)
        
        #comb = torch.concat([feat, edges], dim=0)
        x = global_add_pool(feat,batch)
        out = self.linear_node(x)
        return out

class EGNNetwork(nn.Module):
    def __init__(self, node_f, edge_f, conv_layers=3, **kwargs):
        super().__init__()
        mult = 6
        self.hidden_size = mult*node_f
        self.out_size = (mult//2)*node_f
        self.pos = 3
        self.conv_layers = conv_layers

        self.enn_layers = nn.ModuleList([\
            dgl.nn.pytorch.conv.EGNNConv(node_f, self.hidden_size, self.out_size, edge_f)
        ])
        for l in range(conv_layers-1):
            layer = dgl.nn.pytorch.conv.EGNNConv(self.out_size, self.hidden_size, self.out_size, edge_f)
            self.enn_layers.append(layer)

        self.linear = nn.Sequential(
            nn.Linear(self.out_size+self.pos, node_f*(mult//3)),
            nn.ReLU(),
            nn.Linear(node_f*(mult//3), 1)
        )

    def forward(self, data):
        batch, feat, edge_index, coord, edge_attr = (\
            data.batch, data.x, data.edge_index, data.pos, data.edge_attr
        )
        g = dgl.graph((edge_index[0], edge_index[1]))
        
        for l in range(self.conv_layers):
            feat, coord = self.enn_layers[l](g, feat, coord, edge_attr)
            feat = torch.nn.functional.relu(feat)
        
        comb = torch.concat([feat, coord], dim=1)
        x = global_add_pool(comb,batch)
        out = self.linear(x)
        return out

class NNConvNetwork(nn.Module):
    def __init__(self, node_f, edge_f, dp_rate=0.1, **kwargs):
        hidden_size = 90
        super().__init__()
        conv1_net = nn.Sequential(
            nn.Linear(edge_f, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, node_f*hidden_size)
        )
        conv2_net = nn.Sequential(
            nn.Linear(edge_f, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*hidden_size)
        )
        conv3_net = nn.Sequential(
            nn.Linear(edge_f, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size*(hidden_size//2))
        )
        self.conv1 = NNConv(node_f, hidden_size, conv1_net)
        self.conv2 = NNConv(hidden_size, hidden_size, conv2_net)
        self.conv3 = NNConv(hidden_size, hidden_size//2, conv3_net)
        self.fc_1 = nn.Linear(hidden_size//2, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, data):
        batch, x, edge_index, coord, edge_attr = (data.batch, data.x, data.edge_index, data.pos, data.edge_attr)
        x = torch.concat([x, coord], dim=1)
        # First graph conv layer
        x = torch.nn.functional.relu(self.conv1(x, edge_index, edge_attr))
        # Second graph conv layer
        x = torch.nn.functional.relu(self.conv2(x, edge_index, edge_attr))
        x = torch.nn.functional.relu(self.conv3(x, edge_index, edge_attr))
        x = global_add_pool(x,batch)
        x = torch.nn.functional.relu(self.fc_1(x))
        output = self.out(x)
        return output

