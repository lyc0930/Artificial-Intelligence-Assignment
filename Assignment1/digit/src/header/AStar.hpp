#ifndef ASTAR_HPP
#define ASTAR_HPP
#include "Board.hpp"
#include <map>
#include <queue>
#include <string>
#include <vector>
namespace AStar
{
    extern std::priority_queue<Node> frontier;
    extern std::map<std::vector<unsigned char>, Step> explored;
    Node GraphSearch(Node, Node);
    std::string Movement(std::vector<unsigned char>, std::vector<unsigned char>);
} // namespace AStar
#endif // !ASTAR_HPP
