#ifndef BOARD_HPP
#define BOARD_HPP
#include <vector>

extern std::vector<char> Direction[25];

class Node
{
public:
    std::vector<unsigned char> Position;
    unsigned short depth;

    Node(std::vector<unsigned char>);

    Node(Node, int, int);

    void print();

    int ManhattanDistance() const;

    bool operator<(const Node &) const;

    bool operator==(const Node &) const;
};

struct Step
{
    unsigned short depth;
    int manhattanDistance;
    unsigned char sliderIndex;
    char direction;
};

#endif // !BOARD_HPP