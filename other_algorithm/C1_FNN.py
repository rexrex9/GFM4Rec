from other_algorithm import Zcommon
from dataloader import dataloader
import torch
from torch import nn
from data_set import filepaths as fp



class FNN( nn.Module ):

    def __init__( self, n_users, item_df, dim ):
        super( FNN, self ).__init__()

        self.item_df = item_df

        self.users = nn.Embedding( n_users, dim,max_norm=1)
        self.item_features = nn.Embedding( item_df.values.max()+1, dim,max_norm=1)
        self.mlp_layer = self.__mlp(dim)

    def __mlp( self, dim ):
        return nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, 1),
            nn.Sigmoid()
        )
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

    # forFMseries
    def getAllFeatures(self, u, i):
        item_feature_indexes = torch.LongTensor(self.item_df.loc[i.cpu()].values).to(Zcommon.device)
        user_feats = torch.unsqueeze(self.users(u), dim=1)
        item_feats = self.item_features(item_feature_indexes)
        all = torch.cat([user_feats, item_feats], dim=1)
        # [batch_size, n_features, dim]
        return all

    def forward(self, u, i):
        all_feature_embs = self.__getAllFeatures( u, i )
        # [batch_size, dim]
        out = self.FMaggregator( all_feature_embs )
        # [batch_size, 1]
        out = self.mlp_layer(out)
        # [batch_size]
        out = torch.squeeze(out)
        return out

def train( data_set_name, epochs = 20, batchSize = 1024, lr = 0.01, dim = 32, need_eva = True ):
    users, items, train_set, test_set = \
        dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING, test_ratio=0.1)
    item_df = dataloader.readItem(fp.DataSet_Dict[data_set_name].ITEMS )

    net = FNN( max(users)+1,item_df,dim )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr, weight_decay=0.5)

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'FNN')

if __name__ == '__main__':
    train('mlLatest')
    #0.71