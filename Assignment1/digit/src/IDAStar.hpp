#ifndef IDASTAR_HPP
#define IDASTAR_HPP
#include "Board.hpp"
#include <map>
#include <set>
#include <string>
#include <vector>
namespace IDAStar
{
    extern unsigned int limit;
    extern std::vector<Node> path;
    extern std::map<std::vector<unsigned char>, Step> explored;
    std::vector<Node> GraphSearch(Node, Node);
    unsigned int RecursiveSearch(unsigned int);
    std::string Movement(std::vector<unsigned char>, std::vector<unsigned char>);
} // namespace IDAStar
#endif // !IDASTAR_HPP