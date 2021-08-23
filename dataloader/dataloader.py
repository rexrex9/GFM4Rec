
from data_set import filepaths as fp
import pandas as pd
import numpy as np
import torch
import networkx as nx

np.random.seed=1

def readRecData(path,test_ratio=0.1):
    df = pd.read_csv(path)
    triples = df.values
    user_set = set(triples[:,0])
    item_set = set(triples[:,1])
    test_indices = np.random.choice(len(triples), int(len(triples) * test_ratio), replace=False)
    train_indices = np.array(list(set(range(len(triples))) - set(test_indices)))
    train_set = triples[train_indices]
    test_set = triples[test_indices]

    # 返回用户集合列表，物品集合列表，与用户，物品，评分三元组列表
    return list(user_set), list(item_set), [tuple(t) for t in train_set], [tuple(t) for t in test_set]

def readItem(item_path):
    item_df = pd.read_csv( item_path, index_col = 0 )
    return item_df

def readGraphData(item_path):
    #user_graph_pairs = np.load(user_path)
    item_graph_pairs = np.load(item_path)
    return item_graph_pairs

##########################Graph#####################################################

# 根据边集生成networkx的图
def get_graph(pairs):
    G = nx.Graph()  # 初始化无向图
    G.add_edges_from(pairs)  # 通过边集加载数据
    return G

def getTorchGraph(pairs):
    edges = torch.tensor([pairs[:,0], pairs[:,1]], dtype=torch.long)
    return edges

# 传入图与物品索引得到torch形式的边集
def graphSage4Rec( G, items, n_sizes = [ 10 ] ):
    '''
    :param G: networkx的图结构数据
    :param items: 每一批次得到的物品索引
    :param n_size: 每次采样的邻居数
    :param n_deep: 采样的深度或者说阶数
    :return: torch.tensor类型的边集
    '''
    leftEdges = [ ]
    rightEdges = [ ]

    for size in n_sizes:
        # 初始的节点指定为传入的物品，之后每次的初始节点为前一次采集到的邻居节点
        target_nodes = list( set ( items ) )
        items = set( )
        for i in target_nodes:
            neighbors = list( G.neighbors( i ) )
            if len(neighbors) >= size: #如果邻居数大于指定个数则仅采样指定个数的邻居节点
                neighbors = np.random.choice( neighbors, size=size, replace = False )
            else:  # 如果邻居数小于指定个数则有放回的随机抽取指定个数的邻居
                neighbors = np.random.choice(neighbors, size=size, replace=True)
            rightEdges.extend( neighbors )
            leftEdges.extend( [ i for _ in neighbors ] )
            # 将邻居节点存下以便采样下一阶邻居时提取
            items |= set( neighbors )
    edges = torch.tensor( [ leftEdges, rightEdges ], dtype = torch.long )
    return edges

# 传入图与物品索引得到dataframe形式的邻居索引集
def graphSage4RecAdjType( G, items, n_sizes = [ 10, 5 ] ):
    '''
    :param G: networkx的图结构数据
    :param items: 每一批次得到的物品索引
    :param n_sizes: 采样的邻居节点数量列表，列表长度为采样深度或者理解为采样阶数。
    为了包方便后续并行计算，每一阶的邻居数量需要保持一致，但不同阶的邻居数量不需要保持一致。
    '''
    adj_lists = [ ]
    for size in n_sizes:
        # 初始的节点指定为传入的物品，之后每次的初始节点为前一次采集到的邻居节点
        target_nodes = items
        neighbor_nodes = []
        items = set( )
        for i in target_nodes:
            neighbors = list( G.neighbors( i ) )
            if len(neighbors) >= size: # 如果邻居数大于指定个数则无放回的随机抽取指定个数的邻居
                neighbors = np.random.choice( neighbors, size = size, replace = False )
            else: # 如果邻居数小于指定个数则有放回的随机抽取指定个数的邻居
                neighbors = np.random.choice( neighbors, size = size, replace = True )
            neighbor_nodes.append( neighbors )
            items |= set( neighbors )
        # 将目标节点与它们的邻居节点索引用DataFrame的数据结构表示，并记录与一个列表中
        adj_lists.append( pd.DataFrame( neighbor_nodes, index = target_nodes ) )

    # 因为消息传递是从外向内，所以将列表倒叙使得外层在前，内层在后。
    adj_lists.reverse( )
    return adj_lists

if __name__ == '__main__':
    pass




if __name__ == '__main__':
    users,items,train_set,test_set = readRecData( fp.Ml_latest.RATING )
    #user_df,item_df = readUserAndItem( fp.Ml_latest.USERS, fp.Ml_latest.ITEMS )

    #print(user_df.values.max())
    #print(item_df.values.max())

    df = pd.read_csv(fp.Ml_latest.RATING)
    n = df.values
    print(type(n))
    print(df)
    print(n)