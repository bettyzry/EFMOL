def getmax(list, topk):
    import pandas as pd
    index = [i for i in range(len(list))]
    df = pd.Series(data=list, index=index)
    df = df.sort_values(ascending=False)
    maxlist = df[:topk].index
    return maxlist

def GetTopNMax(xf_score, topk):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    maxnindex = []
    for score in xf_score:
        max_num_index = getmax(list(score), topk)
        maxnindex.append(max_num_index)
    return maxnindex



