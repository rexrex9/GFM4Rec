
from dataloader import dataloader
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from data_set import filepaths as fp
from other_algorithm import Zcommon

class GCN4Rec( torch.nn.Module ):

    def __init__( self, n_users, n_item_features, dim, hidden_dim ,G):
        super( GCN4Rec, self ).__init__( )

        self.users = nn.Embedding( n_users, dim,max_norm=1)
        self.item_features = nn.Embedding( n_item_features, dim,max_norm=1)

        self.all_item_indices = torch.LongTensor( range( n_item_features ) ).to(Zcommon.device)

        self.i_conv1 = GCNConv( dim, hidden_dim )
        self.i_conv2 = GCNConv( hidden_dim, dim )

        self.G = G

    def gnnForwardItem( self, e, edges):
        # [ n_entitys, dim ]
        x = self.item_features( self.all_item_indices )
        # [ n_entitys, hidden_dim ]
        x = F.dropout(  F.relu( self.i_conv1( x, edges ) ) )
        # [ n_entitys, dim ]
        x = self.i_conv2( x, edges )
        # 通过物品的索引取出[ batch_size, dim ]形状的张量表示该批次的物品
        return x[e]

    def forward( self, u, i ):
        i_index = i.cpu().detach().numpy()
        i_edges = dataloader.graphSage4Rec(self.G, i_index).to(Zcommon.device)

        # [ batch_size, dim ]
        users = self.users( u )
        items = self.gnnForwardItem( i, i_edges )
        # [batch_size ]
        uv = torch.sum( users * items, dim=1 )
        # [batch_size ]
        logit = torch.sigmoid( uv )
        return logit

def train(data_set_name,epochs=20,batchSize=1024,dim=128,hidden_dim=64,lr=0.002,need_eva=True):
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING)
    # 读取所有节点索引及表示物品全量图的边集对
    item_graph_pairs = dataloader.readGraphData(fp.DataSet_Dict[data_set_name].ITEM_GRAPH)

    G = dataloader.get_graph( item_graph_pairs )
    net = GCN4Rec( max(user_set)+1, item_graph_pairs.max()+1, dim, hidden_dim ,G)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=0.5)

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'GCN4Rec')
    return net


if __name__ == '__main__':
    train('mlLatest')
