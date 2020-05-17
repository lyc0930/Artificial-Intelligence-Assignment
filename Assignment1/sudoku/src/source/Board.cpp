#include "../header/Board.hpp"
#include <algorithm>
#include <array>
#include <iostream>
#include <set>

Board::Board(){};

Board::Board(std::array<std::array<int, 10>, 10> b) : board(b) {}

Board B;

int &Board::operator()(int x, int y)
{
    return board[x][y];
}

int &Board::operator[](Location l)
{
    return board[l.x][l.y];
}

void Board::print()
{
    for (int i = 1; i <= 9; i++)
    {
        for (int j = 1; j <= 9; j++)
            std::cout << board[i][j] << ' ';
        std::cout << std::endl;
    }
    return;
}

bool Board::BackTrackingSolve(Location l)
{
    if (l)
    {
        if ((*this)[l] != 0)
            l++;
        for (int choice : l.choices())
        {
            (*this)[l] = choice;
            if (BackTrackingSolve(l.next()))
                return true;
            (*this)[l] = 0;
        }
        return false;
    }
    else
        return true;
}

void Board::operator=(std::array<std::array<int, 10>, 10> b)
{
    board = b;
}

Location::Location(int x, int y) : x(x), y(y) {}

Location::Location(const Location &l) : x(l.x), y(l.y) {}

Location Location::next() const
{
    Location nextLocation(this->x, this->y);
    do
    {
        nextLocation.y++;
        if (nextLocation.y > 9)
        {
            nextLocation.y = 1;
            nextLocation.x++;
            if (nextLocation.x > 9)
            {
                nextLocation.x = -1;
                nextLocation.y = -1;
                break;
            }
        }
    } while (B[nextLocation] != 0);
    return nextLocation;
}

Location Location::operator++()
{
    Location originLocation(this->x, this->y);
    do
    {
        y++;
        if (y > 9)
        {
            y = 1;
            x++;
            if (x > 9)
            {
                x = -1;
                y = -1;
                break;
            }
        }
    } while (B[*this] != 0);
    return originLocation;
}

Location Location::operator++(int n)
{
    do
    {
        y++;
        if (y > 9)
        {
            y = 1;
            x++;
            if (x > 9)
            {
                x = -1;
                y = -1;
                break;
            }
        }
    } while (B[*this] != 0);
    return this->next();
}

Location::operator bool() const
{
    if (x < 1 || x > 9 || y < 1 || y > 9)
        return false;
    return true;
}

std::set<int> Location::row_choices() const
{
    std::set<int> choices = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 1; i <= 9; i++)
        choices.erase(B(x, i));
    return choices;
}
std::set<int> Location::column_choices() const
{
    std::set<int> choices = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 1; i <= 9; i++)
        choices.erase(B(i, y));
    return choices;
}
std::set<int> Location::grid_choices() const
{
    std::set<int> choices = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 1; i <= 3; i++)
        for (int j = 1; j <= 3; j++)
            choices.erase(B((x - 1) / 3 * 3 + i, (y - 1) / 3 * 3 + j));
    return choices;
}
std::set<int> Location::choices() const
{
    std::set<int> _choices, choices;
    std::set<int> RowChoices = row_choices(), ColumnChoices = column_choices(), GridChoices = grid_choices();
    std::set_intersection(RowChoices.begin(), RowChoices.end(), ColumnChoices.begin(), ColumnChoices.end(), inserter(_choices, _choices.begin()));
    std::set_intersection(_choices.begin(), _choices.end(), GridChoices.begin(), GridChoices.end(), inserter(choices, choices.begin()));
    if (x == y)
        for (int i = 1; i <= 9; i++)
            choices.erase(B(i, i));
    if (x + y == 10)
        for (int i = 1; i <= 9; i++)
            choices.erase(B(i, 10 - i));
    return choices;
}