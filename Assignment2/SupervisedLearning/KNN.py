import numpy as np
from rich.progress import (
    BarColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
)  # 进度条


def distanceBetween(j, q):
    '''
    ## 计算向量间 Minkowski 距离
    ### Arguments
    - `j` 向量 $x_{j}$
    - `q` 向量 $x_{q}$

    ### Returns
    - 向量间 Minkowski 距离 $(\sum_{i}|x_{j,i} - x_{q,i}|^p)^{1/p}$
        - $p = 2$ 欧几里得距离
        - $p = 1$ 曼哈顿距离
    '''
    # return np.sqrt(np.sum(np.square(j - q)))  # p = 2 : Euclidean metric
    return np.sum(np.abs(j - q))  # p = 1 : Manhattan distance


def NearestNeighbor(trainData, trainLabel, testDatum, K):
    '''
    ## 通过k-最近邻确定测试数据的标签
    ### Arguments
    - `trainData` 训练数据集
    - `trainLabel` 训练标签集
    - `testDatum` 测试数据样本
    - `K` 最近邻样本数目

    ### Returns
    - 预测标签
    '''

    Distances = np.array([distanceBetween(trainDatum, testDatum)
                          for trainDatum in trainData])  # 向量间距离

    topK_Neighbors = np.argpartition(Distances, K)[:K]  # k-近邻

    labelList = [0, 0]  # 0 : failed  1 : passed
    for x in topK_Neighbors:
        labelList[int(trainLabel[x])] += 1  # 统计标签为对应类别的近邻数
    return labelList.index(max(labelList))  # 返回具有最多相同近邻数的标签


def predict(trainData, trainLabel, testData, K):
    '''
    ## 测试模型正确率
    ### Arguments
    - `trainData` 训练集数据集
    - `trainLabel` 训练集标记
    - `testData` 测试集数据集
    - `K` 选择近邻数

    ### Returns
    - `predictLabel` 预测标签
    '''
    predictLabel = []
    with Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.completed}/{task.total}",
        "•",
        TimeRemainingColumn(),
    ) as progress:  # rich 进度条
        testTask = progress.add_task(
            "[cyan]predicting...", total=len(testData))
        for x in testData:
            predictLabel.append(NearestNeighbor(
                trainData, trainLabel, x, K))  # 预测标签分类
            progress.update(testTask, advance=1)
    return predictLabel
