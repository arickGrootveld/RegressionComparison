from utilities import regressionLinesSimulation

regressionLinesSimulation(targFuncOrder=5, polyOrders=[1,3], numTrainSamps=100, numTestSamps=100, plotLines=False, printTableOfRegMSEs=True, x_dist='uniform', x_ub=2, x_lb=-2, rngSeed=-1)


