# 原文链接：https://blog.csdn.net/Gamer_gyt/article/details/51113753

'''
import fpGrowth
simpDat = fpGrowth.loadSimpDat()
initSet = fpGrowth.createInitSet(simpDat)
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)
myFPtree.disp()

import fpGrowth
dataSet = fpGrowth.loadSimpDat()
freqItems = fpGrowth.fpGrowth(dataSet)
freqItems
'''

class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}
 
    def inc(self, numOccur):
        self.count += numOccur
 
    def disp(self, ind=1):
        print (' ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet, minSup=1):
    ''' 创建FP树 '''
    # 第一次遍历数据集，创建头指针表
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 移除不满足最小支持度的元素项
    
    # for k in headerTable.keys(): python 2 version
    for k in list(headerTable):
        if headerTable[k] < minSup:
            del(headerTable[k])
    # 空元素集，返回空
    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None
    # 增加一个数据项，用于存放指向相似元素项指针
    for k in headerTable:
        headerTable[k] = [headerTable[k], None]
    retTree = treeNode('Null Set', 1, None) # 根节点
    # 第二次遍历数据集，创建FP树
    for tranSet, count in dataSet.items():
        localD = {} # 对一个项集tranSet，记录其中每个元素项的全局频率，用于排序
        for item in tranSet:
            if item in freqItemSet:
                localD[item] = headerTable[item][0] # 注意这个[0]，因为之前加过一个数据项
        if len(localD) > 0:
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p: p[1], reverse=True)] # 排序
            updateTree(orderedItems, retTree, headerTable, count) # 更新FP树
    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:
        # 有该元素项时计数值+1
        inTree.children[items[0]].inc(count)
    else:
        # 没有这个元素项时创建一个新节点
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        # 更新头指针表或前一个相似元素项节点的指针指向新节点
        if headerTable[items[0]][1] == None:
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
 
    if len(items) > 1:
        # 对剩下的元素项迭代调用updateTree函数
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLink != None):
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat
 
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict

def findPrefixPath(basePat, treeNode):
    ''' 创建前缀路径 '''
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: str(p[1]))]
    '''
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1])] 此处报错
    TypeError: ‘<’ not supported between instances of ‘treeNode’ and 'treeNode’

    解决办法有2种：
    方法一：将p[1]转换成str类型
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:str(p[1]))]

    方法二：将p[1]改成p[1][0]
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p:p[1][0])]
    明确指定比较的元素是第一列，如果相等则按照原有顺序排列。
    ————————————————
    版权声明：本文为CSDN博主「Janice18」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
    原文链接：https://blog.csdn.net/qq_25886325/article/details/90548833
    '''
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup)
 
        if myHead != None:
            # 用于测试
            print ('conditional tree for:', newFreqSet)
            myCondTree.disp()
 
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

def fpGrowth(dataSet, minSup=3):
    initSet = createInitSet(dataSet)
    myFPtree, myHeaderTab = createTree(initSet, minSup)
    freqItems = []
    mineTree(myFPtree, myHeaderTab, minSup, set([]), freqItems)
    return freqItems