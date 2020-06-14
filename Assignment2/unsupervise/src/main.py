import numpy as np
import matplotlib.pyplot as plt
import csv


def loadData(file):
    '''
    加载数据集
    ========
    Arguments
    ---------
    - `file` 数据集文件

    Returns
    -------
    - `Data` 数据集
    - `Identifiers` 类别标签
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


def saveData(data, file):
    '''
    保存聚类后的数据
    ========
    Arguments
    ---------
    - `data` 聚类后数据
    - `file` 文件目录

    Returns
    -------
    '''

    print('start writeing ' + file)
    with open(file, "w") as fileStream:
        csvWriter = csv.writer(fileStream)
        csvWriter.writerows(data)


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

    a = [0] * data.shape[0]
    b = [0] * data.shape[0]
    Silhouette = [0] * data.shape[0]
    for i in range(data.shape[0]):
        a[i] = np.mean([distanceBetween(data[i], data[j]) for j in range(
            data.shape[0]) if i != j and cluster[i] == cluster[j]])  # i 到其簇中其他点距离的均值

        minNeighborDistance = float("inf")
        minNeighborCentroid = 0
        for j in range(k):
            if (distance:=distanceBetween(data[i], centroids[j])) < minNeighborDistance and j != cluster[i]: # 计算观测到质心距离
                minNeighborDistance = distance
                minNeighborCentroid = j
        b[i] = np.mean([distanceBetween(data[i], data[j]) for j in range(
            data.shape[0]) if cluster[j] == minNeighborCentroid])  # i 到相邻簇中所有点距离的均值
        Silhouette[i] = (b[i] - a[i]) / max(a[i], b[i])

    data_clustered = np.insert(
        data, 0, values=cluster + 1, axis=1)  # 以首列的正整数表示类别

    return data_clustered, np.mean(Silhouette)


def clusterTest(trueLabel, clusterLabel):
    '''
    评价聚类效果
    ===========
    Arguments
    ---------
    - `trueLabel` 实际标签
    - `clusterLabel` 聚类标签

    Formula
    -------
        (a + d) / (a + b + c + d)
        - a 为在 trueLabel 中属同一类且在 clusterLabel 中也属同一类的数据点对数
        - b 为在 trueLabel 中属同一类但在 clusterLabel 中不属同一类的数据点对数
        - c 为在 trueLabel 中不属同一类但在 clusterLabel 中属同一类的数据点对数
        - d 为在 trueLabel 中不属同一类且在 clusterLabel 中也不属同一类的数据点对数

    Returns
    -------
    - 兰德系数(Rand index, RI)
    '''

    a = b = c = d = 0
    for i in range(len(trueLabel)):
        for j in range(i + 1, len(trueLabel)):  # 遍历数据点对
            if trueLabel[i] == trueLabel[j]:  # 在 trueLabel 中属同一类
                if clusterLabel[i] == clusterLabel[j]:  # 在 clusterLabel 中也属同一类
                    a += 1
                else:  # 在 clusterLabel 中不属同一类
                    b += 1
            else:  # 在 trueLabel 中不属同一类
                if clusterLabel[i] == clusterLabel[j]:  # 在 clusterLabel 中属同一类
                    c += 1
                else:  # 在 clusterLabel 中也不属同一类
                    d += 1
    print('a = {:5}  b = {:5}'.format(a, b))
    print('c = {:5}  d = {:5}'.format(c, d))

    return (a + d) / (a + b + c + d)  # 兰德系数


if __name__ == "__main__":
    Data, Identifiers = loadData('../data/wine/wine.data')  # 读取数据与实际类别

    silhouetteCoefficient = []
    for k in range(1, 13):
        data_clustered, silhouette = KMeans(k, PCA(Data, 0.99))
        silhouetteCoefficient.append(silhouette)

    plt.bar(list(range(1, 13)), silhouetteCoefficient, align='center')
    plt.title('Silhouette Graph')
    plt.xlabel('k-cluster')
    plt.ylabel('Silhouette Coefficient')
    plt.savefig('../output/SilhouetteCoefficient.png')  # 显示类别数与轮廓系数关系

    data_clustered, silhouette = KMeans(3, PCA(Data, 0.99))
    saveData(data_clustered, '../output/wine_clustered.csv')  # 聚类后结果保存至 csv 文件
    print('Rand index = ', clusterTest(Identifiers, data_clustered[:, 0]))
