// #include "AStar.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <vector>
using namespace std;
string Color[21] = {"47;30",
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
vector<int> Direction[25] =
    {
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

template <
    class T,
    class Container = std::vector<T>,
    class Compare = std::less<typename Container::value_type>>
class MyQueue : public std::priority_queue<T, Container, Compare>
{
public:
    typedef typename std::priority_queue<
        T,
        Container,
        Compare>::container_type::const_iterator const_iterator;

    const_iterator find(const T &val) const
    {
        auto first = this->c.cbegin();
        auto last = this->c.cend();
        while (first != last)
        {
            if ((*first) == val)
                return first;
            ++first;
        }
        return last;
    }
    const_iterator begin() const
    {
        return this->c.cbegin();
    }
    const_iterator end() const
    {
        return this->c.cend();
    }
};
class Node
{
public:
    vector<unsigned char> Position;
    unsigned short depth;

    Node(vector<unsigned char> initialState) : Position(initialState), depth(0) {}

    Node(Node parent, int index, int offset)
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

    void print()
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
                if (Position[i * 5 + j] == 0)
                    cout << "   ";
                else
                {
                    // string formatString = "\033[" + Color[Position[i * 5 + j] - 1] + "m%3d\033[0m";
                    string formatString = "%3d";
                    printf(formatString.c_str(), (int)(Position[i * 5 + j]));
                }
            cout << endl;
        }
        cout << endl;
        return;
    }

    int ManhattanDistance()
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

    friend bool operator<(Node a, Node b)
    {
        if (a.depth + a.ManhattanDistance() > b.depth + b.ManhattanDistance())
            return true;
        else
            return a.depth > b.depth;
    }

    friend bool operator==(Node a, Node b)
    {
        return a.Position == b.Position;
    }
};

int main()
{
    // Node start({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 0, 0, 21});
    Node start({2, 3, 11, 4, 5, 0, 8, 14, 9, 10, 0, 7, 7, 12, 13, 1, 15, 7, 16, 18, 6, 19, 20, 17, 21});
    // Node start({1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 14, 15, 7, 12, 10, 0, 0, 11, 17, 13, 19, 20, 16, 21, 18}); // 2
    Node goal({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0});

    MyQueue<Node> frontier;
    frontier.push(start);
    set<vector<unsigned char>> explored;
    int l = 0;
    while (!frontier.empty())
    {
        Node node = frontier.top();
        frontier.pop();
        cout << l++ << ':' << endl;
        node.print();
        cout << node.depth << endl;
        // getchar();
        if (node == goal)
        {
            cout << node.depth << "goal" << endl;
            break;
        }
        explored.insert(node.Position);
        for (int i = 0; i < 25; i++)
        {
            if (node.Position[i] != 0)
            {
                if (node.Position[i] != 7)
                {
                    for (int d : Direction[i])
                        if (node.Position[i + d] == 0)
                        {
                            Node child(node, i, d);
                            // child.print();
                            auto existNodePointer = frontier.find(child);
                            if ((existNodePointer == frontier.end()) && (explored.count(child.Position) == 0))
                            {
                                frontier.push(child);
                                explored.insert(child.Position);
                            }
                        }
                }
                else
                {
                    if (node.Position[i + 1] == 7) // 首个 7
                    {
                        int d = 0;
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
                        // child.print();
                        auto existNodePointer = frontier.find(child);
                        if ((existNodePointer == frontier.end()) && (explored.count(child.Position) == 0))
                        {
                            frontier.push(child);
                            explored.insert(child.Position);
                        }
                    }
                }
            }
        }
        // getchar();
    }

    return 0;
}