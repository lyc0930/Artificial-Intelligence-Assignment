import numpy as np
from rich.progress import (
    BarColumn,
    TimeRemainingColumn,
    Progress,
    TaskID,
)  # 进度条


class SupportVectorMachine:
    '''
    ## 支持向量机
    ### Methods
    - `Kernel(j, k)` 计算核函数
    - `train()` 训练模型
    '''

    def __init__(self, sigma=10, C=200, epsilon=0.0001):
        '''
        ## 类构造函数
        ### Arguments
        - `sigma` 高斯核参数
        - `C` 软间隔惩罚参数
        - `epsilon` 松弛变量
        '''
        self.__sigma = sigma
        self.__C = C
        self.__epsilon = epsilon

    def Kernel(self, j, k):
        '''
        ## 核函数
        ### Arguments
        - `j` 数据点 $x_j$
        - `k` 数据点 $x_k$

        ### Returns
        - 核函数值
        '''
        # return self.__LinearKernel(j, k)
        return self.__GaussianKernel(j, k)

    def __LinearKernel(self, j, k):
        '''
        # 线性核
        # Arguments
        - `j` 数据点 $x_j$
        - `k` 数据点 $x_k$
        # Formula
        - 向量内积
        # Returns
        - 核函数值
        '''

        return np.dot(j, k)

    def __GaussianKernel(self, j, k, sigma=None):
        '''
        ## 高斯核函数
        ### Arguments
        - `j` 数据点 $x_j$
        - `k` 数据点 $x_k$
        - `sigma`

        ### Formula
            $$
                exp(-||x_j - x_k||^2 / 2 sigma^2)
            $$
        - 如果 sigma 选得很大的话，高次特征上的权重实际上衰减得非常快，所以实际上（数值上近似一下）相当于一个低维的子空间；反过来，如果 sigma 选得很小，则可以将任意的数据映射为线性可分——当然，这并不一定是好事，因为随之而来的可能是非常严重的过拟合问题。

        ### Returns
        - 核函数值
        '''
        if sigma == None:
            sigma = self.__sigma

        return np.exp(-np.sum(np.square(j - k)) / (2 * sigma**2))

    def __ifSatisfyKKT(self, i, C=None, epsilon=None):
        '''
        ## 判断训练样本点$(x_i, y_i)$是否违反 KKT 条件
        ### Arguments
        - `x`
        - `y`
        - `i`
        - `C`
        - `epsilon` 松弛变量

        ### Returns
        训练样本点$(x_i, y_i)$是否符合 KKT 条件
        '''
        if C == None:
            C = self.__C
        if epsilon == None:
            epsilon = self.__epsilon

        z = self.__y[i] * (np.sum([self.__alpha[j] * self.__y[j] * self.__K[i, j]
                                   for j in range(self.__alpha.size)]) + self.__b)  # y_i * g(x_i)

        if ((-epsilon < self.__alpha[i] < epsilon) and (z >= 1 - epsilon)) or \
            ((C - epsilon < self.__alpha[i] < C + epsilon) and (z <= 1 + epsilon)) or \
                ((-epsilon < self.__alpha[i] < C + epsilon) and (1 - epsilon <= z <= 1 + epsilon)):
            return True
        return False

    def __Error(self, i):
        '''
        ## 计算误差项
        ### Arguments
        - `i`

        ### Formula
        - $E_i = g(x_i) - y_i$

        ### Returns
        - $E_i$
        '''

        return np.sum([self.__alpha[j] * self.__y[j] * self.__K[i, j]
                       for j in range(self.__alpha.size)]) + self.__b - self.__y[i]

    def train(self, trainData, trainLabel):
        '''
        ## 训练
        ### Arguments
        - `trainData` 训练集数据
        - `trainLabel` 训练集标签

        ### Algorithm
        - Sequential minimal optimization, SMO

        ### Returns
        '''
        self.__x, self.__y = np.array(trainData), np.array(trainLabel)
        self.__alpha = np.zeros(self.__x.shape[0])
        self.__b = 0
        self.__K = np.zeros(
            [self.__x.shape[0], self.__x.shape[0]], dtype=float)  # 训练数据核函数表

        for i in range(self.__x.shape[0]):
            for j in range(i, self.__x.shape[0]):
                self.__K[i, j] = self.__K[j, i] = self.Kernel(
                    self.__x[i], self.__x[j])  # 计算核函数表

        with Progress(
            "[progress.description]{task.description}",
            BarColumn(bar_width=None),
            "[progress.percentage]{task.completed}/{task.total}",
            "•",
            "[progress.remaining]{task.elapsed:.2f}",
        ) as progress:  # rich 进度条
            allSatisfied = False  # 全部满足 KKT 条件
            while not allSatisfied:
                allSatisfied = True
                iterateTask = progress.add_task(
                    "[cyan]iterating...", total=self.__x.shape[0])
                for i in range(self.__x.shape[0]):  # 外层循环
                    progress.update(iterateTask, advance=1)
                    if not (self.__ifSatisfyKKT(i)):  # 选择第一个变量
                        E1 = self.__Error(i)
                        maximum = -1
                        for k in range(self.__x.shape[0]):  # 内层循环
                            tempE = self.__Error(k)
                            tempE_difference = np.fabs(E1 - self.__Error(k))
                            if tempE_difference > maximum:  # 选择第二个变量
                                maximum = tempE_difference
                                E2 = tempE
                                j = k
                        if maximum == -1:
                            continue

                        U = max(0, (self.__alpha[i] + self.__alpha[j] - self.__C) if self.__y[i]
                                == self.__y[j] else (self.__alpha[j] - self.__alpha[i]))  # alpha^2_new 的下界
                        V = min(self.__C, (self.__alpha[i] + self.__alpha[j]) if self.__y[i]
                                == self.__y[j] else (self.__alpha[j] - self.__alpha[i] + self.__C))  # alpha^new_2 的上界
                        alpha_2_new = self.__alpha[j] + self.__y[j] * (E1 - E2) / (
                            self.__K[i, i] + self.__K[j, j] - 2 * self.__K[i, j])

                        # alpha^2_new 越界
                        if alpha_2_new > V:
                            alpha_2_new = V
                        elif alpha_2_new < U:
                            alpha_2_new = U

                        alpha_1_new = self.__alpha[i] + self.__y[i] * \
                            self.__y[j] * (self.__alpha[j] - alpha_2_new)

                        # 更新偏置
                        b_1_new = -E1 - self.__y[i] * self.__K[i, i] * (
                            alpha_1_new - self.__alpha[i]) - self.__y[j] * self.__K[j, i] * (alpha_2_new - self.__alpha[j]) + self.__b
                        b_2_new = -E2 - self.__y[i] * self.__K[i, j] * (
                            alpha_1_new - self.__alpha[i]) - self.__y[j] * self.__K[j, j] * (alpha_2_new - self.__alpha[j]) + self.__b

                        # 实装更新
                        if (np.fabs(self.__alpha[i] - alpha_1_new) < 0.0000001) and (np.fabs(self.__alpha[j] - alpha_2_new) < 0.0000001):
                            continue
                        else:
                            allSatisfied = False

                        self.__alpha[i] = alpha_1_new
                        self.__alpha[j] = alpha_2_new

                        if 0 < alpha_1_new < self.__C:
                            self.__b = b_1_new
                        elif 0 < alpha_2_new < self.__C:
                            self.__b = b_2_new
                        else:
                            self.__b = (b_1_new + b_2_new) / 2

                progress.stop_task(iterateTask)
        return

    def classify(self, testDatum):
        '''
        ## 预测测试数据的标签
        ### Arguments
        - `testDatum` 测试数据

        ### Returns
        - 分类决策函数值
        '''
        distance = np.sum([self.__alpha[i] * self.__y[i] * self.Kernel(testDatum, self.__x[i])
                           for i in range(self.__x.shape[0]) if self.__alpha[i] > 0]) + self.__b  # 支持向量 alpha > 0
        return np.sign(distance)


if __name__ == '__main__':
    SVM = SupportVectorMachine()
    SVM.train([[3, 3], [4, 3], [1, 1]], [1, 1, -1])
    print(SVM.classify([0, 4]))
