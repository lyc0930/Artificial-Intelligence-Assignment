#ifndef ASTAR_H
#define ASTAR_H
#include <vector>
class Node
{
public:
    std::vector<unsigned char> Position;
    unsigned short depth;

    Node(std::vector<unsigned char> initialState);

    Node(Node parent, int index, int offset);

    void print();

    friend bool operator<(Node a, Node b);

    friend bool operator==(Node a, Node b);
};

#endif