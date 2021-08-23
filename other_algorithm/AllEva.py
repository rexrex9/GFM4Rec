from data_set import filepaths as fp
from dataloader import dataloader
import os
import torch
from other_algorithm import Zcommon
from tqdm import tqdm
#datas = ['mlLatest','ml1m','Bx']
datas = ['ml1m','ml10m','mlLatest']

model_dir = fp.Model.T2
report_path =fp.Report.T2EVA

def readTestData():
    print('读取数据')
    dataSets = {}
    for d in tqdm(datas):
        _, _, _, test_set = dataloader.readRecData(path=fp.DataSet_Dict[d].RATING)
        dataSets[d]=test_set
    return dataSets

def do():
    dataSets = readTestData()
    with open(report_path,'w+') as f:
        for one_path in os.listdir(model_dir):
            name = one_path.replace('.model','')
            print(name)
            setname = name.split('_')[1]
            net = torch.load(os.path.join(model_dir,one_path))
            auc, f1 = Zcommon.doEva(net, dataSets[setname], batchSize=1024)
            f.write('{}\tauc:{}\tF1:{}\n'.format(name,auc,f1))

if __name__ == '__main__':
    do()

