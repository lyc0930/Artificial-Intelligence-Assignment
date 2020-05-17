#ifndef ASTAR_HPP
#define ASTAR_HPP
#include "Board.hpp"
#include <map>
#include <queue>
#include <string>
#include <vector>
namespace AStar
{
    // 搜索边沿
    extern std::priority_queue<Node> frontier;
    // 已探索结点及移动信息
    extern std::map<std::vector<unsigned char>, Step> explored;
    // 图搜索最优解
    Node GraphSearch(Node, Node);
    // 起终结点间的移动序列
    std::string Movement(std::vector<unsigned char>, std::vector<unsigned char>);
} // namespace AStar
#endif // !ASTAR_HPP
