from major import GAFM_user,GAFM_base,GAFM_item
from other_algorithm import A1_LR, A2_ALS, B1_FM, B2_AFM, C1_FNN, C2_Deep_FM, D2_GAT4Rec


#datas = ['mlLatest','ml1m','Bx']
datas = ['ml10m' ]
#Models = [ GAFM_user ]

Models = [A1_LR, A2_ALS, B1_FM, B2_AFM, C1_FNN, C2_Deep_FM, D2_GAT4Rec, GAFM_base,GAFM_item]

def doOne( Model ):
    print( Model )
    for d in datas:
        batch = 20480 if d=='ml10m' else 1024
        print( d )
        Model.train( d, epochs = 10, batchSize=batch, need_eva = False )

def do( ):
    for one in Models:
        doOne( one )

if __name__ == '__main__':
    do( )