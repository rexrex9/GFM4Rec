import pandas as pd
from dataloader import dataloader
import torch
from torch import nn
from data_set import filepaths as fp
from other_algorithm import Zcommon


class GAT4Rec( torch.nn.Module ):

    def __init__( self, n_users, n_entitys, dim ,G):

        super( GAT4Rec, self ).__init__( )

        self.entitys = nn.Embedding(n_entitys, dim, max_norm=1)
        self.users = nn.Embedding(n_users, dim, max_norm=1)

        self.multiHeadNumber = 2

        self.W = nn.Linear( in_features = dim, out_features = dim//self.multiHeadNumber, bias = False )
        self.a = nn.Linear( in_features = dim, out_features = 1, bias = False)
        self.leakyRelu = nn.LeakyReLU( negative_slope = 0.2 )

        self.G = G
    def oneHeadAttention( self,target_embeddings, neighbor_entitys_embeddings ):
        # [ batch_size, w_dim ]
        target_embeddings_w = self.W( target_embeddings )
        # [ batch_size, n_neighbor, w_dim ]
        neighbor_entitys_embeddings_w = self.W( neighbor_entitys_embeddings )
        # [ batch_size, n_neighbor, w_dim ]
        target_embeddings_broadcast = torch.cat(
            [ torch.unsqueeze( target_embeddings_w, 1)
             for _ in range( neighbor_entitys_embeddings.shape[1] ) ], dim = 1 )
        # [ batch_size, n_neighbor, w_dim*2 ]
        cat_embeddings = torch.cat( [ target_embeddings_broadcast, neighbor_entitys_embeddings_w ], dim=-1 )
        # [ batch_size, n_neighbor, 1 ]
        eijs = self.leakyRelu( self.a( cat_embeddings ) )
        # [ batch_size, n_neighbor, 1 ]
        aijs = torch.softmax( eijs, dim = 1 )
        # [ batch_size, w_dim]
        out = torch.sum( aijs * neighbor_entitys_embeddings_w, dim = 1 )
        return out

    def multiHeadAttentionAggregator( self, target_embeddings, neighbor_entitys_embeddings ):
        '''
        :param target_embeddings: 目标节点的向量 [ batch_size, dim ]
        :param neighbor_entitys_embeddings: 目标节点的邻居节点向量 [ batch_size, n_neighbor, dim ]
        '''
        embs = []
        for i in range( self.multiHeadNumber ): # 循环多头注意力的头数
            embs.append( self.oneHeadAttention( target_embeddings, neighbor_entitys_embeddings ) )
        # 将每次单头注意力层得到的输出张量拼接后输出
        return torch.cat( embs, dim=-1 )


    def forward( self, u,i ):
        i_index = i.cpu().detach().numpy()
        i_edges = dataloader.graphSage4RecAdjType(self.G, i_index)
        # [batch_size, dim]
        items = self.gnnForward( i_edges )
        # [batch_size, dim]
        users = self.users(u)
        # [batch_size]
        uv = torch.sum(users * items, dim=1)
        # [batch_size]
        logit = torch.sigmoid(uv)
        return logit


def train(data_set_name,epochs=20,batchSize=1024,dim=128,lr=0.002,need_eva=True):
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING)
    item_graph_pairs = dataloader.readGraphData(fp.DataSet_Dict[data_set_name].ITEM_GRAPH)
    G = dataloader.get_graph( item_graph_pairs )

    net = GAT4Rec(max(user_set)+1, item_graph_pairs.max()+1,dim,G)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=0.5)

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'GAT4Rec')


if __name__ == '__main__':
    train('mlLatest')

    #0.8575