import treePlotter
import _trees
import trees
import treeplotlib
# dataset, labels = _trees.createdataset()
# # print(trees.calcshannonent(dataset))
# # splitresult = trees.splitdataset(dataset, 1, 1)
# # print(splitresult)
# # bestfeature = trees.choosebestfeaturetosplit(dataset)
# # print(bestfeature)
# mytree = trees.creattree(dataset, labels)
# print(mytree)

# treeplotlib.createPlot()

# 在这计算叶子数和层数测试

# mytree1 = trees.retrieveTree(1)
# # leafnum = trees.getNumLeafs(mytree1)
# # treesdepth = trees.getTreeDepth(mytree1)
# #
# # print(f"leafnum: {leafnum} \ntreesdepth: {treesdepth}")
# #
# # treePlotter.createPlot(mytree1)
# # mytree1['no surfacing'][3] = 'maybe'
# # treePlotter.createPlot(mytree1)
#
# result1 = trees.classify(trees.retrieveTree(1), labels, [1, 1, 1])
# result2 = trees.classify(trees.retrieveTree(1), labels, [1, 0, 1])
# result3 = trees.classify(trees.retrieveTree(1), labels, [0, 0, 1])
#
# print(result1, result2, result3)
#

# 使用pickle模块读取和存储树

# trees.storetree(mytree1, 'classifier.txt')
# tree = trees.grabtree('classifier.txt')
# print(tree)

# 使用决策树预测隐形眼镜类型
f = open('lenses.txt')
# 数据格式处理
lenses = [inst.strip().split('\t') for inst in f.readlines()]
lenseslabels = ['age', 'prescript', 'astigmatic', 'tearrate']
# 训练，决策树生成
lensestree = trees.creattree(lenses, lenseslabels)
trees.storetree(lensestree, 'lenseclassifier.txt')
print(lensestree)
treeplotlib.createPlot(lensestree)

