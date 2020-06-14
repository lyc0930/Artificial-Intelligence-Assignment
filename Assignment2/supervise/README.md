# 人工智能基础编程作业 2

## 监督学习——学生表现预测

### 数据集

[Student Performance Data Set](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

### 属性说明

数据集是包含了学生的各种属性，详见[数据集说明](https://github.com/lyc0930/Artificial-Intelligence-Assignment/blob/master/Assignment2/supervise/data/student/student.txt)，学习目标是根据其他属性（包含成绩` G1``G2 `与否）预测成绩`G3`

### 学习算法

-   KNN, k-NearestNeighbor k-近邻算法
-   SVM, Support Vector Machine 支持向量机
-   LR, Logistic Regression 逻辑斯蒂回归

### 文件结构

```sh
├───data
│   └───student
│           student-mat.csv
│           student-merge.R
│           student-por.csv
│           student.txt
│
└───src
        KNN.py
        LR.py
        main.py
        SVM.py
```

### 依赖

程序依赖于`numpy`及`sklearn`（仅`sklearn.preprocessing`），并使用`rich`包显示了训练与预测过程的进度条

### 运行

在`src`目录下运行`main.py`文件，使用`-h`或`--help`的参数获得如下的帮助

```sh
> python main.py -h
usage: main.py [-h] {KNN,SVM,LR} ...

Simple machine learning test

optional arguments:
  -h, --help    show this help message and exit

Learning Algorithms:
  {KNN,SVM,LR}
    KNN         k-Nearest Neighbors
    SVM         Support Vector Machine
    LR          Logistic Regression

PB17000297 罗晏宸 AI Programming Assignment 2
```

其中学习算法是必需的选项，可以从 k-近邻`KNN`、支持向量机`SVM`、 Logistic 回归`LR`中选择。可以在选择参数后使用`-h`或`--help`的参数获得更多帮助，下面以支持向量机为例

```sh
> python main.py SVM -h
usage: main.py SVM [-h] [-C penalty] [-t xi] {Gaussian,Linear,Polynomial} ...

Support Vector Machine

optional arguments:
  -h, --help            show this help message and exit
  -C penalty            Soft margin penalty hyperparameter for support vector machine (default: 200)
  -t xi, --toler xi     Slack variable (toler) for support vector machine (default: 0.0001)

Kernel Functions:
  {Gaussian,Linear,Polynomial}
    Gaussian            Gaussian kernel function(default)
    Linear              Linear kernel function
    Polynomial          Polynomial kernel function
```

进一步选择高斯核函数后，可以获得更多关于核参数的帮助与默认值信息

```sh
> python main.py SVM Gaussian -h
usage: main.py SVM Gaussian [-h] [-s sigma]

Gaussian kernel function

optional arguments:
  -h, --help            show this help message and exit
  -s sigma, --sigma sigma
                        Parameter of gaussian kernel function for support vector machine (default: 10)
```

其他核函数乃至算法的命令是类似的。

## 结果

程序成功运行后可以在终端看到训练及预测过程的进度，预测完成后会输出如下的模型评价信息

```sh
Elapsed time: 1.869s
TP = 262  TN =  69
FP =  61  FN =   3
F1 score: 89.115646%
```
