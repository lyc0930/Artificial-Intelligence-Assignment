'''
LR(Logistic Regression)
===
    使用 Logistic 回归算法预测分类标签
Provides
--------
- Logistic 回归分类器类::

    >>> lr = LogisticRegressionClassifier(iteration=200, learning_rate=0.0001)

- 在训练集`trainData`及训练集标签`trainLabel`上训练模型::

    >>> lr.train(trainData, trainLabel)

- 预测测试数据`testDatum`的分类标签::

    >>> lr.classify(testDatum)

- 使用训练集`trainData`及训练集标签`trainLabel`，预测测试数据集`testData`的分类标签::

    >>> predict(trainData, trainLabel, testData, iteration=200, learning_rate=0.0001)

'''

import numpy as np
from rich.progress import (
    BarColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
)  # 进度条


class LogisticRegressionClassifier:
    '''
    Logistic 回归分类器
    ========
    Methods
    -------
    - `train(trainData, trainLabel)` 训练模型
    - `classify(testDatum)` 预测类别
    '''

    def __init__(self, iteration=200, learning_rate=0.0001):
        '''
        类构造函数
        ========
        Arguments
        ---------
        - `iteration` 迭代次数
        - `learning_rate` 学习速率
        '''

        self.__alpha = learning_rate
        self.__iteration = iteration

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, trainData, trainLabel):
        '''
        训练
        == =
        Arguments
        ---------
        - `trainData` 训练集数据
        - `trainLabel` 训练集标签

        Returns
        -------
        '''
        self.__x = np.insert(np.array(trainData).astype(
            float), 0, values=1.0, axis=1)  # 训练数据，增加哑变量
        self.__y = np.array(trainLabel)  # 训练样本
        self.__weights = np.zeros(self.__x.shape[1], dtype=float)  # 初始化分类器权重

        progress = Progress(
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeRemainingColumn(),
        )  # rich 进度条
        progress.start()

        trainTask = progress.add_task(
            "[cyan]training...", total=self.__iteration)

        for iter in range(self.__iteration):
            for i in range(self.__x.shape[0]):
                h = self.__sigmoid(np.dot(self.__x[i], self.__weights))
                self.__weights += self.__alpha * \
                    (self.__y[i] - h) * h * (1 - h) * self.__x[i]  # 更新权重
                progress.update(trainTask, advance=1 / self.__x.shape[0])

        progress.stop()
        return

    def classify(self, testDatum):
        '''
        预测测试数据的标签
        ==============
        Arguments
        ---------
        - `testDatum` 测试数据

        Returns
        -------
        - 分类决策函数值
        '''
        h = self.__sigmoid(
            np.dot(self.__weights, np.insert(testDatum, 0, values=1.0)))  # 增加哑变量
        return 1 if h >= 0.5 else 0  # 二分类


def predict(trainData, trainLabel, testData, iteration=200, learning_rate=0.0001):
    '''
    测试模型正确率
    ===========
    Arguments
    ---------
    - `trainData` 训练集数据集
    - `trainLabel` 训练集标记
    - `testData` 测试集数据集
    - `iteration` 迭代次数
    - `learning_rate` 学习速率

    Returns
    -------
    - `predictLabel` 预测标签
    '''
    predictLabel = []
    classifier = LogisticRegressionClassifier(iteration, learning_rate)
    classifier.train(trainData, trainLabel)

    progress = Progress(
        "[progress.description]{task.description}",
        BarColumn(bar_width=None),
        "[progress.percentage]{task.completed}/{task.total}",
        "•",
        TimeRemainingColumn(),
    )  # rich 进度条
    progress.start()

    testTask = progress.add_task(
        "[cyan]predicting...", total=len(testData))

    for testDatum in testData:
        predictLabel.append(classifier.classify(testDatum))  # 预测标签分类
        progress.update(testTask, advance=1)

    progress.stop()
    return predictLabel
