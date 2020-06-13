import numpy as np


def loadData(file):
    '''
    加载数据集
    ========
    Arguments
    ---------
    - `file` 数据集文件

    Returns
    -------
    - `` 数据集
    '''
    print('start reading ' + file)
    Attributes = []
    Identifiers = []
    total = [0 for i in range(13)]  # 各分量总和

    with open(file, 'r') as fileStream:
        for line in fileStream.readlines():
            datum = line.strip().split(',')
            Attributes.append([float(x) for x in datum[1:]])
            Identifiers.append(int(datum[0]))
        Data = np.array(Attributes)
        average = np.mean(Data, axis=0)  # 属性平均值
        standardDeviation = np.std(Data, axis=0)  # 属性方差
        Data = (Data - average) / standardDeviation  # Z-Score 标准化
    return Data, Identifiers


def PCA(data, threshold):
    '''
    利用主成分分析对数据矩阵进行降维
    ===
    Arguments
    ---------
    - `data` （Z-Score 标准化后的）数据矩阵
    - `threshold` 特征值的累计贡献率

    Algorithm
    ---------
    - Principal components analysis，PCA

    Formula
    -------
    Sum(first m-1 eigenvalues) / Sum(all eigenvalues) < threshold <= Sum(first m eigenvalues) / Sum(all eigenvalues)

    Returns
    -------
    - `lowerDimensionalData` 降维之后的矩阵数据矩阵
    '''

    eigenValues, eigenVectors = np.linalg.eig(
        np.cov(data, rowvar=0))  # 由协方差矩阵计算特征值与特征向量
    eigenValuesIndices = np.argsort(eigenValues)[::-1]  # 由大到小排序后的下标
    total = np.sum(eigenValues)  # 总和
    total_m = 0
    for m in range(len(eigenValues)):
        if total_m / total >= threshold:
            break
        total_m += eigenValues[eigenValuesIndices[m]]
    # 选取前 m 个特征值对应的特征向量，作为新的特征空间的一组基
    eigenVectors_m = eigenVectors[eigenValuesIndices[0:m]]
    lowerDimensionalData = np.dot(data, eigenVectors_m.T)  # 原始数据乘以基实现降维

    return lowerDimensionalData


if __name__ == "__main__":
    Data, Identifiers = loadData('../data/wine/wine.data')
    print(Data)
    print(Identifiers)
    print(PCA(Data, 0.8))
