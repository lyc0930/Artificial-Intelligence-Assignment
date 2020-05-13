#include "Board.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

std::vector<char> Direction[25] = {
    {1, 5},
    {1, 5, -1},
    {1, 5, -1},
    {1, 5, -1},
    {5, -1}, // 第 1 行
    {-5, 1, 5},
    {-5, 1, 5, -1},
    {-5, 1, 5, -1},
    {-5, 1, 5, -1},
    {-5, -1, 5}, // 第 2 行
    {-5, 1, 5},
    {-5, 1, 5, -1},
    {-5, 1, 5, -1},
    {-5, 1, 5, -1},
    {-5, -1, 5}, // 第 3 行
    {-5, 1, 5},
    {-5, 1, 5, -1},
    {-5, 1, 5, -1},
    {-5, 1, 5, -1},
    {-5, -1, 5}, // 第 4 行
    {-5, 1},
    {-5, 1, -1},
    {-5, 1, -1},
    {-5, 1, -1},
    {-5, -1} // 第 5 行
};

Node::Node(std::vector<unsigned char> initialState) : Position(initialState), depth(0) {}

Node::Node(Node parent, int index, int offset)
{
    Position = parent.Position;
    if (Position[index] != 7)
    {

        Position[index + offset] = Position[index];
        Position[index] = 0;
    }
    else
    {
        for (int i : {0, 1, 6})
            Position[index + i] = 0;
        for (int i : {0, 1, 6})
            Position[index + i + offset] = 7;
    }
    depth = parent.depth + 1;
}

void Node::print()
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            if (Position[i * 5 + j] == 0)
                std::cout << "   ";
            else
                std::cout << std::setw(3) << (int)(Position[i * 5 + j]);
        std::cout << std::endl;
    }
    return;
}

int Node::ManhattanDistance() const
{
    int m = 0;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            unsigned char element = Position[i * 5 + j];
            if (element == 0)
                continue;
            if (element == 6)
                m += abs(i - 2) + j;
            else if (element == 7)
            {
                if (Position[i * 5 + j + 1] == 7)
                    m += abs(i - 1) + j;
            }
            else if (element <= 10)
                m += abs(i - (element - 1) / 5) + abs(j - (element - 1) % 5);
            else
                m += abs(i - (element + 1) / 5) + abs(j - (element + 1) % 5);
        }
    }
    return m;
}

bool Node::operator<(const Node &that) const
{
    if (this->depth + this->ManhattanDistance() == that.depth + that.ManhattanDistance())
        return this->depth > that.depth;
    else
        return this->depth + this->ManhattanDistance() > that.depth + that.ManhattanDistance();
}

bool Node::operator==(const Node &that) const
{
    return this->Position == that.Position;
}
