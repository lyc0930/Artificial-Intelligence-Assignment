#include "Board.hpp"
#include <algorithm>
#include <cmath>
#include <ctime>
#include <iostream>
#include <map>
#include <queue>
#include <vector>
using namespace std;

int main()
{
    // Node start({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 21});
    Node start({2, 3, 11, 4, 5, 0, 8, 14, 9, 10, 0, 7, 7, 12, 13, 1, 15, 7, 16, 18, 6, 19, 20, 17, 21}); // 1
    // Node start({1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 14, 15, 7, 12, 10, 0, 0, 11, 17, 13, 19, 20, 16, 21, 18}); // 2
    // Node start({0, 6, 15, 7, 7, 8, 9, 13, 4, 7, 1, 2, 3, 10, 5, 14, 11, 16, 12, 18, 19, 20, 17, 21, 0}); // 3
    Node goal({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0});

    priority_queue<Node> frontier;
    frontier.push(start);
    map<vector<unsigned char>, Step> explored;

    clock_t startTime = clock();

    while (!frontier.empty())
    {
        Node node = frontier.top();
        frontier.pop();
        while (node.Position == frontier.top().Position)
            frontier.pop(); // 处理重复结点
        if (node == goal)
        {
            cout << node.depth << "goal" << endl;
            break;
        }
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
                            if (exist != explored.end() && explored.at(child.Position).depth > child.depth)
                            {
                                explored[child.Position] = Step({child.depth, child.ManhattanDistance(), node.Position[i], d});
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
                        if (exist != explored.end() && explored.at(child.Position).depth > child.depth)
                        {
                            explored[child.Position] = Step({child.depth, child.ManhattanDistance(), node.Position[i], d});
                        }
                    }
                }
            }
        }
    }
    clock_t endTime = clock();
    cout << "运行时间：" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    return 0;
}