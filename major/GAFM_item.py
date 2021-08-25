import pandas as pd
from torch.utils.data import DataLoader
from dataloader import dataloader
from tqdm import tqdm
import torch
from torch import nn
from data_set import filepaths as fp
from other_algorithm import Zcommon

class GAFM_Item( torch.nn.Module ):

    def __init__( self, n_users, n_entitys, k_dim,t_dim,G):

        super( GAFM_Item, self ).__init__( )

        self.entitys = nn.Embedding( n_entitys, k_dim, max_norm = 1 )
        self.users = nn.Embedding( n_users, k_dim, max_norm = 1 )

        self.a_liner = nn.Linear(k_dim, t_dim)
        self.h_liner = nn.Linear(t_dim, 1)

        self.G = G

    #FM聚合
    def FMaggregator( self, feature_embs ):
        # feature_embs:[ batch_size, n_features, k ]
        # [ batch_size, k ]
        square_of_sum = torch.sum( feature_embs, dim = 1 )**2
        # [ batch_size, k ]
        sum_of_square = torch.sum( feature_embs**2, dim = 1 )
        # [ batch_size, k ]
        output = square_of_sum - sum_of_square
        return output

    #注意力计算
    def attention_item( self, target_embs,embs ):
        # target_embs : [batch_size, k]
        # embs: [ batch_size, k ]
        #[ batch_size, t ]
        embs = self.a_liner( target_embs * embs )
        #[ batch_size, t ]
        embs = torch.relu( embs )
        #[ batch_size, 1 ]
        embs = self.h_liner( embs)
        #[ batch_size, 1 ]
        atts = torch.softmax( embs, dim=1 )
        return atts

    # 根据上一轮聚合的输出向量，原始索引与记录原始索引与更新后索引的映射表得到这一阶的输入邻居节点向量
    def __getEmbeddingByNeibourIndex( self, orginal_indexes, nbIndexs, aggEmbeddings ):
        new_embs = []
        for v in orginal_indexes:
            embs = aggEmbeddings[ torch.squeeze( torch.LongTensor( nbIndexs.loc[v].values ).to(Zcommon.device) ) ]
            new_embs.append( torch.unsqueeze( embs, dim = 0 ) )
        return torch.cat( new_embs, dim = 0 )

    def gnnForward( self, adj_lists ):
        n_hop = 0
        for df in adj_lists:
            if n_hop == 0:
                entity_embs = self.entitys(torch.LongTensor(df.values).to(Zcommon.device))
            else:
                entity_embs = self.__getEmbeddingByNeibourIndex(df.values, neighborIndexs, aggEmbeddings)
            target_embs = self.entitys(torch.LongTensor(df.index).to(Zcommon.device))
            if n_hop < len( adj_lists ):
                neighborIndexs = pd.DataFrame( range( len( df.index ) ), index = df.index )
            aggEmbeddings = self.FMaggregator(entity_embs )
            # [batch_size, dim]
            atts = self.attention_item(target_embs, aggEmbeddings)
            aggEmbeddings = atts * aggEmbeddings + target_embs
        # 返回最后的目标节点向量也就是指定代表这一批次的物品向量,形状为 [ batch_size, dim ]
            n_hop+=1
        return aggEmbeddings

    def forward( self, u,i ):
        i_index = i.cpu().detach().numpy()
        adj_lists = dataloader.graphSage4RecAdjType(self.G, i_index)
        # [batch_size, dim]
        items = self.gnnForward( adj_lists )
        # [batch_size, dim]
        users = self.users(u)
        # [batch_size]
        uv = torch.sum(users * items, dim=1)
        # [batch_size]
        logit = torch.sigmoid(uv)
        return logit

def train(data_set_name,epochs=20,batchSize=1024,k_dim=128,t_dim=64,lr=0.002,need_eva=True):
    user_set, item_set, train_set, test_set = dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING)
    item_graph_pairs = dataloader.readGraphData(fp.DataSet_Dict[data_set_name].ITEM_GRAPH)
    G = dataloader.get_graph( item_graph_pairs )

    net = GAFM_Item(max(user_set)+1,item_graph_pairs.max()+1,k_dim,t_dim,G)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr,weight_decay=0.5)

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'GAFMItem')






if __name__ == '__main__':
    train('mlLatest')