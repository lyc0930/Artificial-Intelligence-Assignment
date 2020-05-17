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

具体解的每步操作以用 `(number, direction)` 表示为`output/DataNumber.txt`中的一行，`number`$\in [1,21]$，`direction`$\in \{\texttt{u},\texttt{d},\texttt{l},\texttt{r}\}$，分别代表将标号为 `number` 的数字块上移、下移、左移、右移。**每个移动序列独占一行**（实验原要求为移动序列之间用英文分号分隔，但考虑到可读性，改为每个行动一行，不影响实验内容）。如`output/1.txt`中的内容如下

```bash
(1,u); (1,u); (6,u); (19,l); (15,d); (7,l); (14,d); (11,d); (3,r); (2,r); (1,u); (14,d); (11,d); (8,r); (7,u); (6,u); (14,l); (14,l); (16,l); (15,u); (20,l); (17,u); (21,l); (21,l)
```

注意，对同一组数据进行求解会覆盖相应的输出文件，且不同算法得到的解有行动顺序上的差异，注意保存运行结果。

## 数独问题

### 问题内容

使用 CSP 问题的回溯搜索(Backtracking Search)算法解决了如下 X 数独问题

> ：在 $9 \times 9$格的方格中，需要根据已知数字，推理出所有剩余空格的数字，并满足每一行、每一列、每一个粗线宫$3 \times 3$内的数字均含 1-9，不重复。九宫格的两条对角线内的数字也均含 1-9，不重复。

算法使用了 MRV 进行优化

### 文件结构

```bash
sudoku
├───input
│       sudoku01.txt
│       sudoku02.txt
│       sudoku03.txt
│
├───output
│       sudoku01.txt
│       sudoku02.txt
│       sudoku03.txt
│
└───src
    │   Makefile
    │   sudoku
    │   sudoku.cpp
    │   sudoku.exe
    │
    ├───header
    │       Board.hpp
    │       fileIO.hpp
    │
    └───source
            Board.cpp
            fileIO.cpp
```

### 编译

`Makefile`文件中已设置好编译链接命令，运行如下的命令

```bash
make     # 默认编译，使用 MRV 优化后的回溯搜索
make all # 同上
make dump # 使用未经优化的简单回溯搜索
```

或直接使用 g++ 对程序进行编译

```bash
g++ source/fileIO.cpp source/Board.cpp sudoku.cpp -O3 [-D MRV] -o sudoku
```

其中若省略预编译参数`MRV`则采用简单回溯搜索。

### 运行

编译后在`sudoku/`目录下以如下命令运行`sudoku`可执行文件

```bash
./sudoku DataNumber
```

`DataNumber`是指定测试数据的参数，程序将使用`input/sudoku0DataNumber.txt`中的数据作为初始状态进行搜索。参数都是必须的，没有默认值。如

```bash
./sudoku 1
```

表示对`input/sudoku01.txt`中的数据作为初始状态进行 X 数独的求解。

## 结果

程序成功运行后可以在终端看到类似下方的运行结果

```bash
运行时间：0.007s
```

给出了程序的运行时间。

数独的解以和输入数据相同的形式存取`output/sudoku0DataNumber.txt`中。如`output/sudoku01.txt`中的内容如下

```bash
2 8 1 5 3 4 9 6 7
5 6 4 9 7 8 1 3 2
7 9 3 1 6 2 4 8 5
9 2 5 7 1 6 8 4 3
4 1 7 3 8 5 6 2 9
6 3 8 2 4 9 7 5 1
3 4 9 6 2 1 5 7 8
8 5 2 4 9 7 3 1 6
1 7 6 8 5 3 2 9 4
```

注意，对同一组数据进行求解会覆盖相应的输出文件，且不同算法得到的解有所不同，注意保存运行结果。
