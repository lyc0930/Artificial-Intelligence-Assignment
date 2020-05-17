#include "../header/AStar.hpp"
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
std::priority_queue<Node> AStar::frontier;
std::map<std::vector<unsigned char>, Step> AStar::explored;

Node AStar::GraphSearch(Node start, Node goal)
{
    using namespace AStar;
    frontier.push(start);
    while (!frontier.empty())
    {
        Node node = frontier.top();
        frontier.pop();
        while (node.Position == frontier.top().Position)
            frontier.pop(); // 处理重复结点
        if (node == goal)
            return node;
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
                            auto exist = explored.find(child.Position);
                            frontier.push(child);
                            if (exist == explored.end() || explored.at(child.Position).depth > child.depth)
                            {
                                explored[child.Position] = Step({child.depth, child.ManhattanDistance(), (unsigned char)i, d});
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
                        auto exist = explored.find(child.Position);
                        frontier.push(child);
                        if (exist == explored.end() || explored.at(child.Position).depth > child.depth)
                        {
                            explored[child.Position] = Step({child.depth, child.ManhattanDistance(), (unsigned char)i, d});
                        }
                    }
                }
            }
        }
    }
    return start;
}

std::string AStar::Movement(std::vector<unsigned char> initialState, std::vector<unsigned char> finalState)
{
    using namespace AStar;
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