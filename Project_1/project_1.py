# -*-coding:utf-8-*-
# https://blog.csdn.net/Gamer_gyt/article/details/51113753

#from pydoc import apropos

#=========================     准备函数 （下）      ==========================================
#加载数据集
def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def createC1(dataSet):
    C1 = []   #C1为大小为1的项的集合
    for transaction in dataSet:  #遍历数据集中的每一条交易
        for item in transaction: #遍历每一条交易中的每个商品
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    #map函数表示遍历C1中的每一个元素执行forzenset，frozenset表示“冰冻”的集合，即不可改变
    return map(frozenset,C1)

#Ck表示数据集，D表示候选集合的列表，minSupport表示最小支持度
#该函数用于从C1生成L1，L1表示满足最低支持度的元素集合
def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            #issubset：表示如果集合can中的每一元素都在tid中则返回true  
            if can.issubset(tid):
                #统计各个集合scan出现的次数，存入ssCnt字典中，字典的key是集合，value是统计出现的次数
                if not (can in ssCnt):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float( len(list(D)) )
    print('numItems: ', numItems)
    retList = []
    supportData = {}
    for key in ssCnt:
        #计算每个项集的支持度，如果满足条件则把该项集加入到retList列表中
        support = ssCnt[key]/numItems
        if support >= minSupport:
            retList.insert(0, key)
        #构建支持的项集的字典
        supportData[key] = support
    return retList,supportData
#====================                准备函数（上）              =============================

#======================          Apriori算法（下）               =================================
#Create Ck,CaprioriGen ()的输人参数为频繁项集列表Lk与项集元素个数k，输出为Ck
def aprioriGen(Lk,k):
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1,lenLk):
            #前k-2项相同时合并两个集合
            L1 = list(Lk[i])[:k-2]
            L2 = list(Lk[j])[:k-2]
            L1.sort()
            L2.sort()
            if L1 == L2:
                retList.append(Lk[i] | Lk[j])

    return retList

def apriori(dataSet, minSupport=0.1):
    C1 = createC1(dataSet)  #创建C1
    D= [set([1, 3, 4]), set([2, 3, 5]), set([1, 2, 3, 5]), set([2, 5])]
    #D = map(set,dataSet)
    L1,supportData = scanD(D, C1, minSupport)
    L = [L1]
    #若两个项集的长度为k - 1,则必须前k-2项相同才可连接，即求并集，所以[:k-2]的实际作用为取列表的前k-1个元素
    k = 2
    while(len(L[k-2]) > 0):
        Ck = aprioriGen(L[k-2], k)
        Lk,supK = scanD(D,Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k +=1
    return L, supportData
#======================          Apriori算法(上)               =================================

if __name__=="__main__":
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet)
    i = 0
    for one in L:
        print ( "项数为 %s 的频繁项集：" % (i + 1), one,"\n")
        i +=1