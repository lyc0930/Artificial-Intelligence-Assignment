#include "AStar.h"
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
std::string Color[21] = {"47;30",
                         "41;37",
                         "42;30",
                         "43;30",
                         "44;32",
                         "45;37",
                         "40;31",
                         "47;31",
                         "41;30",
                         "42;37",
                         "43;31",
                         "44;33",
                         "45;32",
                         "46;30",
                         "47;33",
                         "41;36",
                         "42;34",
                         "43;34",
                         "44;37",
                         "45;36",
                         "46;31"};

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
            Position[i] = 0;
        for (int i : {0, 1, 6})
            Position[i + offset] = 7;
    }
    depth = parent.depth + 1;
}

void Node::print()
{
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
            if (Position[i * 5 + j] == 0)
                cout << "   ";
            else
            {
                string formatString = "\033[" + Color[Position[i * 5 + j] - 1] + "m%3d\033[0m";
                printf(formatString.c_str(), (int)(Position[i * 5 + j]));
            }
        cout << endl;
    }
    cout << endl;
    return;
}

bool Node::operator<(Node a, Node b)
{
    return a.Position[1] > b.Position[1];
}

bool Node::operator==(Node a, Node b)
{
    return a.Position == b.Position;
}