from other_algorithm import Zcommon
from dataloader import dataloader
import torch
from torch import nn
from data_set import filepaths as fp

class AFM( nn.Module ):

    def __init__( self, n_users, item_df, k_dim,t_dim ):
        super( AFM, self ).__init__()


        self.item_df = item_df

        self.users = nn.Embedding( n_users, k_dim,max_norm=1)
        self.item_features = nn.Embedding( item_df.values.max()+1, k_dim,max_norm=1)

        self.a_liner = nn.Linear(k_dim, t_dim)
        self.h_liner = nn.Linear(t_dim, 1)
        self.p_liner = nn.Linear(k_dim, 1)

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
    def attention( self, embs ):
        # embs: [ batch_size, k ]
        #[ batch_size, t ]
        embs = self.a_liner( embs )
        #[ batch_size, t ]
        embs = torch.relu( embs )
        #[ batch_size, 1 ]
        embs = self.h_liner( embs)
        #[ batch_size, 1 ]
        atts = torch.softmax( embs, dim=1 )
        return atts

    # forFMseries
    def getAllFeatures(self, u, i):
        item_feature_indexes = torch.LongTensor(self.item_df.loc[i.cpu()].values).to(Zcommon.device)
        user_feats = torch.unsqueeze(self.users(u), dim=1)
        item_feats = self.item_features(item_feature_indexes)
        all = torch.cat([user_feats, item_feats], dim=1)
        # [batch_size, n_features, dim]
        return all

    def forward( self, u, i ):
        #取出特征向量
        all_feature_embs = self.__getAllFeatures( u, i )
        #经过FM层得到输出
        embs = self.FMaggregator( all_feature_embs )
        #得到注意力
        atts = self.attention( embs )
        #[ batch_size, 1 ]
        outs = self.p_liner(atts * embs)
        #[ batch_size ]
        outs = torch.squeeze(outs)
        # [ batch_size ]
        logit = torch.sigmoid( outs )
        return logit


def train( data_set_name, epochs = 20, batchSize = 1024, lr = 0.01,k_dim=64,t_dim=32, need_eva = True ):
    users, items, train_set, test_set = \
        dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING, test_ratio=0.1)
    item_df = dataloader.readItem(fp.DataSet_Dict[data_set_name].ITEMS)

    net = AFM( max(users)+1 ,item_df,k_dim,t_dim )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr, weight_decay=0.6)

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'AFM')
if __name__ == '__main__':
    train('mlLatest')
    #0.78