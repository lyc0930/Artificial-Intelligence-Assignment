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


def distanceBetween(j, q):
    '''
    计算向量间欧式距离
    ======================
    Arguments
    ---------
    - `j` 向量 $x_{j}$
    - `q` 向量 $x_{q}$

    Formula
    -------
        sqrt(sum_{i}|x_{j,i} - x_{q,i}|^2)

    Returns
    -------
    - 向量间 Minkowski 欧几里得距离
    '''
    return np.sqrt(np.sum(np.square(j - q)))  # Euclidean metric


def KMeans(k, data):
    '''
    使用 k-means 算法将数据进行聚类
    ===
    Arguments
    ---------
    - `k` 聚类数
    - `data` （降维后的）数据矩阵

    Algorithm
    ---------
    - k-means

    Returns
    -------
    - 聚类后的数据
    - 聚类的轮廓系数
    '''

    centroids = np.empty([k, data.shape[1]], dtype=float)  # 簇质心
    for i in range(k):
        index = np.random.randint(data.shape[0])  # 随机下标
        centroids[i] = data[index]  # Forgy 方法：选取随机观测作为初始质心

    isClusteringChanged = True  # 聚类是否改变
    cluster = np.zeros(data.shape[0], dtype=int)   # 类别

    while isClusteringChanged:
        isClusteringChanged = False
        for i in range(data.shape[0]):  # 更新质心
            minDistance = float("inf")
            minCentroid = 0
            for j in range(k):
                if (distance:=distanceBetween(data[i], centroids[j])) < minDistance: # 计算观测到质心距离
                    minDistance = distance
                    minCentroid = j
            if cluster[i] != minCentroid:
                cluster[i] = minCentroid
                isClusteringChanged = True

        for j in range(k):
            centroids[j] = np.mean(
                data[np.nonzero(cluster == j)], axis=0)  # 根据类别更新质心

    return cluster + 1  # 以正整数表示类别


if __name__ == "__main__":
    Data, Identifiers = loadData('../data/wine/wine.data')
    print(KMeans(3, PCA(Data, 0.8)))
