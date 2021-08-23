from dataloader import dataloader
from data_set import filepaths as fp
from torch import nn
import torch
from other_algorithm import Zcommon

class ALS (nn.Module):
    def __init__(self, n_users, n_items, dim):
        super(ALS, self).__init__()
        self.users = nn.Embedding( n_users, dim,max_norm=1 )
        self.items = nn.Embedding( n_items, dim,max_norm=1 )


    def forward(self, u, v):
        print(u.device)
        u = self.users(u)
        v = self.items(v)
        uv = torch.sum( u*v, dim = 1)
        logit = torch.sigmoid(uv)
        return logit


def train( data_set_name,epochs = 10, batchSize = 1024, lr = 0.01, dim = 32, need_eva = True):
    users, items, train_set, test_set = \
        dataloader.readRecData(fp.DataSet_Dict[data_set_name].RATING, test_ratio = 0.1)

    net = ALS(max(users)+1, max(items)+1, dim)
    optimizer = torch.optim.AdamW( net.parameters(), lr = lr, weight_decay=0.5 )
    criterion = torch.nn.BCELoss()

    Zcommon.commonTrain(epochs, net, optimizer, criterion,
                        train_set, test_set, batchSize, need_eva, data_set_name, 'ALS')


if __name__ == '__main__':
    train('mlLatest')

    #0.69