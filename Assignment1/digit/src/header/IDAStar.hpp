#ifndef IDASTAR_HPP
#define IDASTAR_HPP
#include "Board.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>
namespace IDAStar
{
    // 深度限制
    extern unsigned int limit;
    // 搜索路径
    extern std::vector<Node> path;
    // 已探索状态对应转移信息
    extern std::map<std::vector<unsigned char>, Step> explored;
    // 迭代搜索最优解
    std::vector<Node> GraphSearch(Node, Node);
    // 递归搜索
    unsigned int RecursiveSearch();
    // 起终结点之间的移动序列
    std::string Movement(std::vector<unsigned char>, std::vector<unsigned char>);
} // namespace IDAStar
#endif // !IDASTAR_HPP