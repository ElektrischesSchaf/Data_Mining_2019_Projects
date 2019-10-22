import fpGrowth

IBMData=fpGrowth.loadIBMData()
initSet=fpGrowth.createInitSet(IBMData)
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 0.5)
myFPtree.disp()

KaggleData=fpGrowth.loadIBMData()
initSet=fpGrowth.createInitSet(KaggleData)
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 0.5)
myFPtree.disp()