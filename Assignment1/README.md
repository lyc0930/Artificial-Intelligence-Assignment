# 人工智能基础编程作业 1

## 数码问题

### 问题内容

使用 $\text{A}^*$ 搜索算法以及迭代 $\text{A}^*$ 搜索算法解决了如下的问题

> 在一个 $5 \times 5$ 的网格中，23 个小方块写有数字“1-21”，剩下的两个空白方块代表空位，特别地，有三个写有 “7” 的方块被绑定在一起，组成了一个 “7” 型块。与空位上、下、左、右相邻的方块可以移动到空位中，记为一次行动；“7” 型块的行动规则是平凡的，需要两个相应位置的空位
> 现给定初始状态与目标状态，要求获得从初始状态到目标状态的合法移动序列。

算法中使用了**曼哈顿距离(Manhattan Distance)**、**简单加权(Simple Weighted)**的曼哈顿距离、考虑**线性冲突(Linear Conflict)**的曼哈顿距离作为启发式函数。

### 文件结构

```sh
digit
├───input
│       1.txt
│       2.txt
│       3.txt
│
├───output
│       1.txt
│       2.txt
│       3.txt
│
└───src
    │   digit
    │   digit.cpp
    │   digit.exe
    │   Makefile
    │
    ├───header
    │       AStar.hpp
    │       Board.hpp
    │       fileIO.hpp
    │       IDAStar.hpp
    │
    └───source
            AStar.cpp
            Board.cpp
            fileIO.cpp
            IDAStar.cpp
```

### 编译

`Makefile`文件中已设置好编译链接命令，运行如下的命令

```bash
make     # 默认编译，使用考虑线性冲突(Linear Conflict)的曼哈顿距离作为启发式函数
make all # 同上
make SW  # 使用简单加权(Simple Weighted)曼哈顿距离作为启发式函数
make LC  # 使用考虑线性冲突(Linear Conflict)的曼哈顿距离作为启发式函数
make MD  # 使用曼哈顿距离作为启发式函数
```

或直接使用 g++ 对程序进行编译

```bash
g++ source/fileIO.cpp source/AStar.cpp source/IDAStar.cpp source/Board.cpp digit.cpp -O3 [-D HEURISTIC_MACRO] -o digit
```

其中预编译参数`HEURISTIC_MACRO`可选`LINEARCONFLICT`或`SIMPLEWEIGHTED`，不使用参数则采用曼哈顿距离作为启发式函数。

### 运行

编译后在`digit/`目录下以如下命令运行`digit`可执行文件

```bash
./digit Algorithm DataNumber
```

`Algorithm`是指定搜索算法的参数，可选`AStar`或`IDAStar`，分别表示使用 $\text{A}^*$ 搜索算法以及迭代 $\text{A}^*$ 搜索算法；`DataNumber`是指定测试数据的参数，程序将使用`input/DataNumber.txt`中的数据作为初始状态进行搜索。两个参数都是必须的，没有默认值。如

```bash
./digit IDAStar 1
```

表示使用迭代 $\text{A}^*$ 搜索算法对`input/1.txt`中的数据作为初始状态进行求解。

## 结果

程序成功运行后可以在终端看到类似下方的运行结果

```bash
24
运行时间：0.03125s
```

第一行表示算法得到的解的步数（行动次数），第二行给出了程序的运行时间。

具体解的每步操作以用 `(number, direction)` 表示为`output/DataNumber.txt`中的一行，`number`$\in [1,21]$，`direction`$\in \{\texttt{u},\texttt{d},\texttt{l},\texttt{r}\}$，分别代表将标号为 `number` 的数字块上移、下移、左移、右移。移动序列之间用英文分号分隔。如`output/1.txt`中的内容如下

```bash
(1,u); (1,u); (6,u); (19,l); (15,d); (7,l); (14,d); (11,d); (3,r); (2,r); (1,u); (14,d); (11,d); (8,r); (7,u); (6,u); (14,l); (14,l); (16,l); (15,u); (20,l); (17,u); (21,l); (21,l)
```

注意，对同一组数据进行求解会覆盖相应的输出文件，且不同算法得到的解有行动顺序上的差异，注意保存运行结果。

## 数独问题
