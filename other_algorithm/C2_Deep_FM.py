import Zcommon
from dataloader import dataloader
from torch.utils.data import DataLoader
import torch
from torch import nn
from data_set import filepaths as fp
from tqdm import tqdm


class Deep_FM( nn.Module ):

    def __init__( self, n_users, item_df, dim ):
        super( Deep_FM, self ).__init__()

        self.item_df = item_df

        self.users = nn.Embedding( n_users, dim,max_norm=1)
        self.item_features = nn.Embedding( item_df.values.max()+1, dim,max_norm=1)

        total_neigbours = 1 + item_df.shape[1]
        self.mlp_layer = self.__mlp( dim * total_neigbours )

    def __mlp( self, dim ):
        return nn.Sequential(
            nn.Linear( dim, dim // 2 ),
            nn.ReLU( ),
            nn.Linear( dim // 2, dim // 4 ),
            nn.ReLU( ),
            nn.Linear( dim // 4, 1 ),
            nn.Sigmoid( ) )

    def FMcross(self, feature_embs):
        # feature_embs:[ batch_size, n_features, dim ]
        # [batch_size, dim]
        square_of_sum = torch.sum( feature_embs, dim = 1)**2
        # [batch_size, dim]
        sum_of_square = torch.sum( feature_embs**2, dim = 1)
        # [batch_size, dim]
        output = square_of_sum - sum_of_square
        # [batch_size, 1]
        output = torch.sum(output, dim=1, keepdim=True)
        # [batch_size]
        return torch.squeeze(output)

    # DNN部分
    def Deep(self, feature_embs):
        # feature_embs:[ batch_size, n_features, dim ]
        # [ batch_size, total_neigbours * dim ]
        feature_embs = feature_embs.reshape((feature_embs.shape[0], -1))
        # [ batch_size, 1 ]
        output = self.mlp_layer(feature_embs)
        # [ batch_size ]
        return torch.squeeze(output)

    def __getAllFeatures( self,u, i ):
        item_feature_indexes = torch.LongTensor( self.item_df.loc[i].values )
        user_feats = torch.unsqueeze(self.users( u ),dim=1)
        item_feats = self.item_features(item_feature_indexes)
        all = torch.cat( [ user_feats, item_feats ], dim = 1 )
        # [batch_size, n_features, dim]
        return all

    def forward( self, u, i ):
        all_feature_embs = self.__getAllFeatures( u, i )
        # [batch_size]
        fm_out = self.FMcross( all_feature_embs )
        # [batch_size]
        deep_out = self.Deep( all_feature_embs )
        # [batch_size]
        out = torch.sigmoid( fm_out + deep_out )
        return out


def train( data_set_name, epochs = 10, batchSize = 1024, lr = 0.01, dim = 32, need_eva = True):
    users, items, train_set, test_set = \
        dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING, test_ratio=0.1)
    item_df = dataloader.readItem( fp.DataSet_Dict[data_set_name].ITEMS )

    net = Deep_FM( max(users)+1,item_df,dim )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr, weight_decay=0.5)

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'DeepFM')
if __name__ == '__main__':
    train('mlLatest')
