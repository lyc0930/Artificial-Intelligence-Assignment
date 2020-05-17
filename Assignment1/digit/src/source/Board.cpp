#include "../header/Board.hpp"
#include <array>
#include <iomanip>
#include <iostream>
#include <utility>
#include <vector>
std::array<std::vector<char>, 25> Direction = {{
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
}};

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

unsigned int Node::g() const
{
    return depth;
}

int Node::h() const
{
#ifdef SIMPLEWEIGHTED
    int m = 0;

    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            unsigned char element = Position[i * 5 + j];
            if (element == 0)
                continue;
            if (element == 1)
                m += (i + j == 0) ? 0 : (i + j + 2);
            if (element == 2)
                m += (i == 0 && j == 1) ? 0 : (i + abs(j - 1) + 1);
            if (element == 6)
            {
                if (Position[5] == 7) // 7 已复位
                    m += (i == 2 && j == 0) ? 0 : (abs(i - 2) + j + 1);
                else
                    m += abs(i - 2) + j;
            }
            else if (element == 7)
            {
                if (Position[i * 5 + j + 1] == 7) // 只对首个出现的 7 进行计算
                    m += (i == 1 && j == 0) ? 0 : (abs(i - 1) + j + 1);
            }
            else if (element <= 10)
                m += abs(i - (element - 1) / 5) + abs(j - (element - 1) % 5);
            else
                m += abs(i - (element + 1) / 5) + abs(j - (element + 1) % 5);
        }
    }
    return m;
#else
#ifdef LINEARCONFLICT
    return (this->ManhattanDistance() + 2 * this->LinearConflict());
#else
    return this->ManhattanDistance();
#endif
#endif
}

#ifdef SIMPLEWEIGHTED
#else
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
#ifdef LINEARCONFLICT
int Node::LinearConflict() const
{
    std::array<std::pair<int, int>, 22> targetPosition =
        {{std::make_pair(0, 0),
          std::make_pair(0, 0),
          std::make_pair(0, 1),
          std::make_pair(0, 2),
          std::make_pair(0, 3),
          std::make_pair(0, 4),
          std::make_pair(2, 0),
          std::make_pair(1, 1),
          std::make_pair(1, 2),
          std::make_pair(1, 3),
          std::make_pair(1, 4),
          std::make_pair(2, 2),
          std::make_pair(2, 3),
          std::make_pair(2, 4),
          std::make_pair(3, 0),
          std::make_pair(3, 1),
          std::make_pair(3, 2),
          std::make_pair(3, 3),
          std::make_pair(3, 4),
          std::make_pair(4, 0),
          std::make_pair(4, 1),
          std::make_pair(4, 2)}};
    int conflict = 0;
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            unsigned char p = Position[i * 5 + j];
            if (p == 0)
                continue;
            if (targetPosition[p].first != i)
                continue;
            for (int k = j + 1; k < 5; k++)
            {
                unsigned char q = Position[i * 5 + k];
                if (q == 0)
                    continue;
                if (targetPosition[q].first != i)
                    continue;
                if (targetPosition[p].second >= k)
                    conflict++;
            }
        }
    }
    for (int i = 0; i < 5; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            unsigned char p = Position[j * 5 + i];
            if (p == 0)
                continue;
            if (targetPosition[p].second != i)
                continue;
            for (int k = j + 1; k < 5; k++)
            {
                unsigned char q = Position[k * 5 + i];
                if (q == 0)
                    continue;
                if (targetPosition[q].second != i)
                    continue;
                if (targetPosition[p].first >= k)
                    conflict++;
            }
        }
    }
    return conflict;
}
#endif
#endif

bool Node::operator<(const Node &that) const
{
    if (this->g() + this->h() == that.g() + that.h())
        return this->depth > that.depth;
    else
        return this->g() + this->h() > that.g() + that.h();
}

bool Node::operator==(const Node &that) const
{
    return this->Position == that.Position;
}
