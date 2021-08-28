import os
import torch
from sklearn.metrics import roc_auc_score, log_loss, ndcg_score,f1_score
from torch.utils.data import DataLoader
import numpy as np
from data_set import filepaths as fp
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def doEva(net, test_set, batchSize):
    net.to(device)
    net.eval()
    auc= 0
    f1 = 0
    #ll = 0
    #y_preds,y_trues = [],[]
    for u, i, r in DataLoader(test_set, batch_size=batchSize, shuffle=False,drop_last=True):
        u = u.to( device )
        i = i.to( device )
        #r = r.to( device )
        y_pred = net( u, i )
        y_true = r.cpu().detach().numpy()
        y_pred = y_pred.cpu().detach().numpy()
        # y_pred4ll = torch.cat([torch.unsqueeze(1-y_pred, 1), torch.unsqueeze(y_pred, 1)], dim=1).detach().numpy()
        # ll +=log_loss(y_true,y_pred4ll)

        #y_trues.append(y_true.tolist())
        #y_preds.append(y_pred.detach().numpy().tolist())
        auc += roc_auc_score(y_true, y_pred)
        y_pred = np.array([1 if i >= 0.5 else 0 for i in y_pred])
        f1 += f1_score(y_true, y_pred)

    counts = len(test_set) // batchSize
    auc /= counts
    #ll /= counts
    #ngdc = ndcg_score(y_trues,y_preds,k=20)
    f1 /= counts
    return auc,f1


def commonTrain(epochs,net,optimizer,criterion,train_set,test_set,batchSize,need_eva,data_set_name,net_name):
    net.to(device)
    max_auc, max_e = 0, 0
    eva_list = []
    # 开始训练
    for e in range(epochs):
        net.train()
        all_lose = 0
        for u, i, r in tqdm(DataLoader(train_set, batch_size=batchSize, shuffle=True)):
            optimizer.zero_grad()
            r = torch.FloatTensor(r.cpu().detach().numpy())
            u = u.to( device )
            i = i.to( device )
            r = r.to( device )
            logits = net(u, i)
            loss = criterion(logits, r)
            all_lose += loss
            loss.backward()
            optimizer.step()
        print('epoch {},avg_loss={:.4f}'.format(e, all_lose / (len(train_set) // batchSize)))

        if need_eva:
            auc, f1 = showEva(net, train_set, test_set, batchSize)
            eva_list.append((auc, f1))
            if max_auc < auc:
                max_auc = auc
                max_e = e
    if need_eva:
        print(max_e, eva_list[max_e])

    saveModel(net, net_name, data_set_name)


def showEva(net, train_set, test_set, batchSize):
    auc,f1 = doEva(net, train_set, batchSize)
    print('Train: AUC {:.4f}\tF1 {:.4f}'.format(auc,f1))
    auc,f1 = doEva(net, test_set, batchSize)
    print('Test: AUC {:.4f}\tF1 {:.4f}'.format(auc,f1))
    return auc,f1


def saveModel(net,net_name,data_set_name):
    torch.save(net, os.path.join(fp.Model.T2, '{}_{}.model'.format(net_name,data_set_name)))




if __name__ == '__main__':
    pass
