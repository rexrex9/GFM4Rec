import os

ROOT = os.path.split(os.path.realpath(__file__))[0]

class Ml_1m( ):
    # 下载地址：https://grouplens.org/datasets/movielens/1m/
    __BASE = os.path.join( ROOT, 'ml-1m' )
    ORGINAL = os.path.join( __BASE, 'orginal' )
    ORGINAL_USRS = os.path.join(ORGINAL,'users.dat')
    ORGINAL_ITEMS = os.path.join(ORGINAL,'movies.dat')
    ORGINAL_RATING = os.path.join(ORGINAL,'ratings.dat')

    INDEX = os.path.join(__BASE, 'index')
    USERS = os.path.join(INDEX, 'users.csv' )
    ITEMS = os.path.join(INDEX, 'items.csv')
    RATING = os.path.join(INDEX, 'rating.csv')
    USER_GRAPH = os.path.join(INDEX,'user_graph.npy')
    ITEM_GRAPH = os.path.join(INDEX,'item_graph.npy')


class Ml_latest( ):
    # 下载地址：https://grouplens.org/datasets/movielens/latest/
    __BASE = os.path.join( ROOT, 'ml-latest-small' )
    ORGINAL = os.path.join( __BASE, 'orginal')
    ORGINAL_ITEMS = os.path.join(ORGINAL,'movies.csv')
    ORGINAL_RATING = os.path.join(ORGINAL,'ratings.csv')

    INDEX = os.path.join(__BASE, 'index')
    USERS = os.path.join(INDEX, 'users.csv' )
    ITEMS = os.path.join(INDEX, 'items.csv')
    RATING = os.path.join(INDEX, 'rating.csv')
    USER_GRAPH = os.path.join(INDEX,'user_graph.npy')
    ITEM_GRAPH = os.path.join(INDEX,'item_graph.npy')


class Ml_10m( ):
    # 下载地址：https://grouplens.org/datasets/movielens/10m/
    __BASE = os.path.join( ROOT, 'ml-10m' )
    ORGINAL = os.path.join( __BASE, 'orginal' )
    ORGINAL_ITEMS = os.path.join(ORGINAL,'movies.dat')
    ORGINAL_RATING = os.path.join(ORGINAL,'ratings.dat')

    INDEX = os.path.join(__BASE, 'index')
    USERS = os.path.join(INDEX, 'users.csv' )
    ITEMS = os.path.join(INDEX, 'items.csv')
    RATING = os.path.join(INDEX, 'rating.csv')
    USER_GRAPH = os.path.join(INDEX,'user_graph.npy')
    ITEM_GRAPH = os.path.join(INDEX,'item_graph.npy')


class Model( ):
    BASE = os.path.join( ROOT, 'model' )
    T0 = os.path.join(BASE, 't0')
    T1 = os.path.join(BASE, 't1')
    T2 = os.path.join(BASE, 't2')
    T3 = os.path.join(BASE, 't3')
    T4 = os.path.join(BASE, 't4')

class Report():
    BASE = os.path.join(ROOT, 'report')
    T1EVA = os.path.join(BASE, 't1_eva.tsv')
    T2EVA = os.path.join(BASE, 't2_eva.tsv')
    T3EVA = os.path.join(BASE, 't3_eva.tsv')
    T4EVA = os.path.join(BASE, 't4_eva.tsv')

DataSet_Dict = {
    'mlLatest':Ml_latest,
    'ml1m':Ml_1m,
    'ml10m':Ml_10m
}


