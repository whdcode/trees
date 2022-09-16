"""决策树"""
import numpy as np
import operator
import pickle


def calcshannonent(dataset):
    """计算数据集的香农熵"""
    numentries = len(dataset)  # 获得数据集中数据实例个数
    labelcount = {}  # 用来保存各个分类标签下的数量
    for featvec in dataset:
        currentlabel = featvec[-1]  # 获取数据实例的标签
        if currentlabel not in labelcount.keys():  # 将不在字典中的分类标签进行统计
            labelcount[currentlabel] = 0
        labelcount[currentlabel] += 1
    shannonent = 0.0
    for key in labelcount:
        prob = float(labelcount[key]) / numentries  # 计算先验概率，即频率
        shannonent -= prob * np.math.log(prob, 2)  # 计算熵
    return shannonent


def createdataset():
    """创建一个数据集"""
    dataset = [[1, 1, 'yes'], [1, 0, 'no'],
               [1, 1, 'maybe'], [0, 1, 'no'],
               [1, 1, 'yes'], [1, 1, 'yes'],
               [1, 1, 'yes'], [0, 1, 'no'],
               ]
    fealabels = ['no surfacing', 'flippers', 'head']
    return dataset, fealabels


def splitdataset(dataset, axis, value):  # dataset为待划分的数据集，axis为所选择的数据划分目标特征对象索引， value为划分的参考值
    """进行数据划分的操作"""
    retdataset = []  # 返回的已提取所得到的数据集
    for featvec in dataset:
        if featvec[axis] == value:  # 若该数据实例中的目标特征值为参考值则将其提取
            reducefeature = featvec[:axis]
            reducefeature.extend(featvec[axis + 1:])  # 将所抽取的实例特征集中除了目标特征对象的其余部分保留
            retdataset.append(reducefeature)
    return retdataset


def choosebestfeaturetosplit(dataset):
    """根据基于每种特征的分类后数据集的熵选出最佳的目标特征进行分类"""
    numfeatures = len(dataset[0]) - 1  # 获取数据集的特征数量
    baseentropy = calcshannonent(dataset)  # 获取原始数据集的香农熵
    # print(baseentropy)
    bestinfogains = 0.0
    bestfeature = -1
    for i in range(numfeatures):
        featlist = [example[i] for example in dataset]  # 获取数据集dataset中的每一行数据中的第i个元素并存储在特征列表中
        uniquevals = set(featlist)

        newentropy = 0.0  # 设置一个存储划分后数据集香农熵的变量
        for value in uniquevals:  # 这里开始对积基于该种特征分类的各个数据分支信息熵计算并汇总
            subdataset = splitdataset(dataset, i, value)
            prob = len(subdataset) / float(len(dataset))  # 获取这次分类分支的先验概率
            newentropy += prob * calcshannonent(subdataset)
            # print(f"The tropy of {value} of feature{i} is {newentropy} ")
        infogains = baseentropy - newentropy  # 获得熵变化，即增益
        # print(infogains)
        if infogains > bestinfogains:  # 接下来是获取最佳增益并接着第二个特征的分类
            bestinfogains = infogains
            bestfeature = i
    return bestfeature  # 返回数据划分越好的特征索引（增益最大，即数据无序度减小的越多）


def majoritycnt(classlist):  # 针对数据集递归处理完所有特征后，该叶子类标签依旧不一致的问题，采用多数表决决定最终叶子节点标签
    """多数表决的方式决定类标签不唯一的问题"""
    classcount = {}  # 用于计数每个类标签的数目
    for vote in classlist:  # 其中classlist中含有多种类标签
        if vote not in classcount.keys():
            classcount[vote] = 0
        classcount[vote] += 1
    sortedclasscount = sorted(classcount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]


def creattree(dataset, label):
    """通过递归构造树"""
    classlist = [example[-1] for example in dataset]  # 获取类标签列表，这里的dataset格式为包含列表元素的列表
    # 第一个递归停止条件，即数据集的类别具有一致性，这里通过当数据中某种的类标签数量等于总的类标签数目时，说明当前的数据集类别具有一致性
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]
    if len(dataset[0]) == 1:  # 第二个递归停止条件，当已经不满足数据一致性的前提下， 若当前的所有数据特征都遍历了一遍，即此时需要通过多数表决决定节点类别
        return majoritycnt(classlist)
    bestfeat = choosebestfeaturetosplit(dataset)  # 通过信息论的原理选择出最佳的数据划分特征索引
    # print(bestfeat)
    bestfeatlabel = label[bestfeat]
    mytree = {bestfeatlabel: {}}  # 使用字典类型的数据结构存储树的信息，判断节点对应本次划分数据的最佳特征
    del (label[bestfeat])  # 删除本次的数据划分特征标签
    featvalues = [example[bestfeat] for example in dataset]  # 获得数据集的特征值
    uniquevals = set(featvalues)  # 建立该特征的所有特征值集合,集合中有几种特征值，该特征标签节点就有几个分支
    for value in uniquevals:
        sublabels = label[:]
        mytree[bestfeatlabel][value] = creattree(splitdataset(dataset, bestfeat, value), sublabels)
    return mytree


def getTreeDepth(myTree):
    """计算树的层数"""
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth


def getNumLeafs(myTree):
    """计算叶子节点数量"""
    numLeafs = 0
    firstStr = list(myTree.keys())[0]    # 这里的myTree.key()为dict_keys对象，需要转化成列表对象
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[
                    key]).__name__ == 'dict':  # test to see if the nodes are dictonaires, if not they are leaf nodes
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def retrieveTree(i):
    """保存了几个测试用的树"""
    listOfTrees = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                   {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                   ]
    return listOfTrees[i]


def classify(inputtree, featlabels, testvec):
    """使用决策树分类实例"""
    firststr = list(inputtree.keys())[0]
    seconddict = inputtree[firststr]
    featindex = featlabels.index(firststr)
    for key in seconddict.keys():
        if testvec[featindex] == key:
            if type(seconddict[key]).__name__ == 'dict':
                classlabel = classify(seconddict[key], featlabels, testvec)
            else:
                classlabel = seconddict[key]

    return classlabel


def storetree(inputtree, filename):
    """存储决策树"""
    with open(filename, 'wb') as f:
        pickle.dump(inputtree, f, protocol=3)


def grabtree(filename):
    """从文件中读取决策树"""
    with open(filename, 'rb') as f:
        tree = pickle.load(f)
    return tree
