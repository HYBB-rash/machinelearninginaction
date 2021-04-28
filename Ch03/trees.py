
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   trees.py
@Time    :   2021/04/28 15:11:32
@Author  :   hyong 
@Version :   1.0
@Contact :   hyong_cs@outlook.com
'''
# here put the import lib

# %%
from math import log
import operator

def createDataSet():
    """生成数据集

    Returns:
        Tuple[list[list], list[str]]: 数据集元组，第一个数据是数据集，第二个是标签数据
    """
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels

# %%

# ? @Time: 2021/04/28 15:25:43
# * @Desc: 查看生成的数据集
myDat, labels = createDataSet()
myDat

# %%

# ? @Time: 2021/04/28 15:26:03
# * @Desc: 编写辅助函数计算香农熵
def calcShannonEnt(dataSet):
    """计算辅助函数，算香农熵

    Args:
        dataSet (list[list]): 待计算的数据集

    Returns:
        float: 香农熵计算结果
    """
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt

# %%

# ? @Time: 2021/04/28 15:26:29
# * @Desc: 可以看到当前数据集的香农熵
calcShannonEnt(myDat)

# %%

# ? @Time: 2021/04/28 15:25:27
# * @Desc: 香农熵越高，代表混合数据越多。尝试添加一个类数据，观察香农熵的变化
myDat[0][-1] = 'maybe'
myDat
calcShannonEnt(myDat)

# %%

# ? @Time: 2021/04/28 15:25:14
# * @Desc: 划分数据集
def splitDataSet(dataSet, axis, value):
    """axis对应的列符合value条件的数据划分出来，生成一个新的子数据集

    Args:
        dataSet (list[list]): 数据集
        axis (int): 指定数据集的列
        value (NoneType): 目标数据

    Returns:
        [type]: 新的子数据集
    """
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            # ? @Time: 2021/04/28 15:30:51
            # * @Desc: extend和append区别在前者是将元素一个一个放进去，后者是直接将list对象放进去
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# %%

# ? @Time: 2021/04/28 15:37:15
# * @Desc: 重新加载数据
myDat, labels = createDataSet()
myDat

# %%

# ? @Time: 2021/04/28 15:36:52
# * @Desc: 数据集，第二列，选取值为1的子数据集
splitDataSet(myDat, 0, 1)

# %%

# ? @Time: 2021/04/28 15:36:26
# * @Desc: 数据集，第一列，选取值为0的子数据集
splitDataSet(myDat, 0, 0)

# %%

# ? @Time: 2021/04/28 15:36:04
# * @Desc: 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    """计算出最合适作划分的数据列

    Args:
        dataSet (list[list]): 数据集

    Returns:
        int: 返回最合适的数据列的序号
    """
    
    # ? @Time: 2021/04/28 15:39:24
    # * @Desc: 最后一列通常是结果，先排除在外。
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    
    # ? @Time: 2021/04/28 15:40:07
    # * @Desc: 历遍所有属性
    for i in range(numFeatures):
        
        # ? @Time: 2021/04/28 15:40:33
        # * @Desc: 将序号对应的属性值选择出来，一会需要计算
        featList = [example[i] for example in dataSet]
        
        # ? @Time: 2021/04/28 15:41:16
        # * @Desc: 去重，重复的值不要
        uniqueVals = set(featList)
        newEntropy = 0.0
        
        # ? @Time: 2021/04/28 15:41:41
        # * @Desc: 根据现在存在的所有值的种类作计算
        for value in uniqueVals:
            
            # ? @Time: 2021/04/28 15:42:12
            # * @Desc: 根据值来划分新的数据子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            
            # ? @Time: 2021/04/28 15:42:47
            # * @Desc: 计算每种划分的信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        
        # ? @Time: 2021/04/28 15:43:50
        # * @Desc: 计算出最好的信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# %%

# ? @Time: 2021/04/28 15:45:45
# * @Desc: 测试输出结果
myDat, labels = createDataSet()
chooseBestFeatureToSplit(myDat)
# 0

# %%

# ? @Time: 2021/04/28 15:46:23
# * @Desc: 根据数据集，说现在以第0列作为结果划分比较合适
myDat

# %%

# ? @Time: 2021/04/28 15:49:24
# * @Desc: 
def majorityCnt(classList):
    """返回出现次数最多的类名

    Args:
        classList (list): 类列表

    Returns:
        any: 次数出现最多的类名
    """
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

# %%

# ? @Time: 2021/04/28 15:50:16
# * @Desc: 创建树
def createTree(dataSet,labels):
    
    # ? @Time: 2021/04/28 15:50:37
    # * @Desc: 将最后一列单独拿出来。一般是类名列
    classList = [example[-1] for example in dataSet]
    
    # ? @Time: 2021/04/28 15:51:04
    # * @Desc: 如果全是同一个类的话，直接返回吧，这是其中一个递归出口
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    
    # ? @Time: 2021/04/28 15:51:53
    # * @Desc: 判断数据集现在一条数据的长度，如果是1，说明特征用完了但是还没选出来。这时候的出口选定为出现次数最多的类名
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    
    # ? @Time: 2021/04/28 15:54:44
    # * @Desc: 选择出最适合的特征的序号，label是表头，其实就是看看表头是什么一会方便可视化
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    
    # ? @Time: 2021/04/28 15:56:47
    # * @Desc: 建树
    myTree = {bestFeatLabel:{}}
    
    # ? @Time: 2021/04/28 15:57:14
    # * @Desc: 接下来会划分子集，对应的这里不删除，子集编号对应的label表头会对应不上
    del(labels[bestFeat])
    
    # ? @Time: 2021/04/28 15:58:06
    # * @Desc: 对这个选出来的属性的每个可能出现的值，划分出子集后，再继续计算下去
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree

# %%

# ? @Time: 2021/04/28 15:59:30
# * @Desc: 测试建树代码

myDat, labels = createDataSet()
mytree = createTree(myDat, labels)

mytree

# %%
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
