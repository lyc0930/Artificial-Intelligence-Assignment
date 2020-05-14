#include "IDAStar.hpp"
#include <algorithm>
#include <climits>
// #include <iostream>
#include <string>
#include <vector>
unsigned int IDAStar::limit;
std::vector<Node> IDAStar::path;
std::map<std::vector<unsigned char>, Step> IDAStar::explored;

std::vector<Node> IDAStar::GraphSearch(Node start, Node goal)
{
    using namespace IDAStar;
    limit = start.ManhattanDistance();
    path.emplace_back(start);
    while (true)
    {
        // std::cout << limit << std::endl;
        unsigned int t = RecursiveSearch(0);
        if (t == 0)
            return path;
        limit = t;
    }
}

unsigned int IDAStar::RecursiveSearch(unsigned int g)
{
    using namespace IDAStar;
    Node node = path.back();
    unsigned int f = g + (unsigned int)node.ManhattanDistance();
    if (f > limit)
        return f;
    // node.print();
    // std::cout << g << ',' << node.ManhattanDistance() << std::endl;
    // getchar();
    if (node.ManhattanDistance() == 0)
        return 0;
    unsigned int min = UINT_MAX;
    for (int i = 0; i < 25; i++)
    {
        if (node.Position[i] != 0)
        {
            if (node.Position[i] != 7)
            {
                for (char d : Direction[i])
                    if (node.Position[i + d] == 0)
                    {
                        Node child(node, i, d);
                        if (std::find(path.begin(), path.end(), child) == path.end())
                        {
                            path.emplace_back(child);
                            if (explored.find(child.Position) == explored.end() || explored.at(child.Position).depth > child.depth)
                                explored[child.Position] = Step({child.depth, child.ManhattanDistance(), (unsigned char)i, d});
                            unsigned int t = RecursiveSearch(g + 1);
                            if (t == 0)
                                return 0;
                            if (t < min)
                                min = t;
                            path.pop_back();
                        }
                    }
            }
            else
            {
                if (node.Position[i + 1] == 7) // 首个 7
                {
                    char d = 0;
                    if (i > 4 && node.Position[i - 5] == 0 && node.Position[i - 4] == 0)
                        d = -5;
                    else if (i % 5 < 3 && node.Position[i + 2] == 0 && node.Position[i + 7] == 0)
                        d = 1;
                    else if (i < 15 && node.Position[i + 5] == 0 && node.Position[i + 11] == 0)
                        d = 5;
                    else if (i % 5 > 0 && node.Position[i - 1] == 0 && node.Position[i + 5] == 0)
                        d = -1;
                    else
                        continue;
                    Node child(node, i, d);
                    if (std::find(path.begin(), path.end(), child) == path.end())
                    {
                        path.emplace_back(child);
                        if (explored.find(child.Position) == explored.end() || explored.at(child.Position).depth > child.depth)
                            explored[child.Position] = Step({child.depth, child.ManhattanDistance(), (unsigned char)i, d});
                        unsigned int t = RecursiveSearch(g + 1);
                        if (t == 0)
                            return 0;
                        if (t < min)
                            min = t;
                        path.pop_back();
                    }
                }
            }
        }
    }
    return min;
}

std::string IDAStar::Movement(std::vector<unsigned char> initialState, std::vector<unsigned char> finalState)
{
    using namespace IDAStar;
    auto state = finalState;
    std::string sequence = "";
    while (explored.find(state) != explored.end() && state != initialState)
    {
        std::string directionCode;
        switch (explored[state].direction)
        {
        case -5:
            directionCode = "u";
            break;
        case 1:
            directionCode = "r";
            break;
        case 5:
            directionCode = "d";
            break;
        case -1:
            directionCode = "l";
            break;
        }
        sequence = "(" + std::to_string((int)(state[explored[state].sliderIndex + explored[state].direction])) + "," + directionCode + ")\n" + sequence;
        Node currentNode(state);
        Node parentNode(currentNode, explored[state].sliderIndex + explored[state].direction, -explored[state].direction);
        state = parentNode.Position;
    }
    return sequence;
}