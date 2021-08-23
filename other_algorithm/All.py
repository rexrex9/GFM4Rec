from major import GAFM_user,GAFM_base,GAFM_item
from other_algorithm import A1_LR,A2_ALS,B1_FM,B2_AFM,C1_FNN,C2_Deep_FM,D1_GCN4Rec,D2_GAT4Rec


#datas = ['mlLatest','ml1m','Bx']
datas = ['mlLatest']
#Models = [A1_LR,A2_ALS,B1_FM,B2_AFM,C1_FNN,C2_Deep_FM,D1_GCN4Rec,D2_GAT4Rec,GAFM_user,GAFM_base,GAFM_item]
Models = [D1_GCN4Rec,D2_GAT4Rec]
def doOne(Model):
    print(Model)
    for d in datas:
        print(d)
        Model.train(d,epochs=10,need_eva=False)

def do():
    for one in Models:
        doOne(one)


if __name__ == '__main__':
    do()