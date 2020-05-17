#ifndef BOARD_HPP
#define BOARD_HPP
#include <vector>

// 普通数字（除 7 外）在各个位置上的可移动方向
extern std::vector<char> Direction[25];

// 结点类
class Node
{
public:
    // 结点状态，即各位置上数字
    std::vector<unsigned char> Position;
    // 结点深度
    unsigned short depth;
    // 构造与给定状态相同的结点
    Node(std::vector<unsigned char>);
    // 构造由父结点移动给定数字后的子结点，并加深深度
    Node(Node, int, int);
    // 在终端中输出结点状态
    void print();
    // 从根节点到结点的代价
    unsigned int g() const;
    // 到目标结点的估计代价
    int h() const;
    // 与目标结点的曼哈顿距离
    int ManhattanDistance() const;

    bool operator<(const Node &) const;

    bool operator==(const Node &) const;
};

// 结点移动信息
struct Step
{
    // g 即深度
    unsigned int depth;
    // h 估计值
    int h;
    // 移动数字的下标
    unsigned char sliderIndex;
    // 移动方向
    char direction;
};

#endif // !BOARD_HPP