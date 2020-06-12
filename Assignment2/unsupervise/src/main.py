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


if __name__ == "__main__":
    Data, Identifiers = loadData('../data/wine/wine.data')
    print(Data)
    print(Identifiers)
