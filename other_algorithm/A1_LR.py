
from dataloader import dataloader
from data_set import filepaths as fp
from torch import nn
import torch
from other_algorithm import Zcommon


class LR( nn.Module ):
    def __init__( self, n_users, item_df, dim  ):
        super(LR, self).__init__()
        self.item_df = item_df

        self.users = nn.Embedding(n_users, dim, max_norm=1)
        self.item_features = nn.Embedding(item_df.values.max() + 1, dim, max_norm=1)

    # forFMseries
    def __getAllFeatures(self, u, i):
        item_feature_indexes = torch.LongTensor(self.item_df.loc[i.cpu()].values).to(Zcommon.device)
        user_feats = torch.unsqueeze(self.users(u), dim=1)
        item_feats = self.item_features(item_feature_indexes)
        all = torch.cat([user_feats, item_feats], dim=1)
        # [batch_size, n_features, dim]
        return all

    def forward( self, u, i ):
        all_feature_embs = self.__getAllFeatures( u, i)
        logits = torch.sigmoid(torch.sum(torch.sum(all_feature_embs,dim=1),dim=1))
        return logits



def train( data_set_name, epochs = 10, batchSize = 1024, lr = 0.01, dim=16 ,need_eva=True):
    users, items, train_set, test_set = \
        dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING, test_ratio=0.1)
    item_df = dataloader.readItem( fp.DataSet_Dict[data_set_name].ITEMS )

    net = LR( max(users)+1,item_df,dim )
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW( net.parameters(), lr=lr, weight_decay = 0.5)

    Zcommon.commonTrain(epochs, net, optimizer, criterion, train_set, test_set, batchSize, need_eva, data_set_name, 'LR')




if __name__ == '__main__':
    train('mlLatest')

    #0.68