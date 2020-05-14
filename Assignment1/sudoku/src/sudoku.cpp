#include <algorithm>
#include <array>
#include <iostream>
#include <set>
using namespace std;

class Location
{
public:
    int x, y;
    Location(int, int);
    Location(const Location &);
    Location next() const;
    Location operator++();
    Location operator++(int);
    explicit operator bool() const;
    set<int> row_choices() const;
    set<int> column_choices() const;
    set<int> grid_choices() const;
    set<int> choices() const;
};

class Board
{
public:
    array<array<int, 10>, 10> board;
    Board();
    Board(array<array<int, 10>, 10>);
    int &operator()(int, int);
    int &operator[](Location);
    void operator=(array<array<int, 10>, 10>);
    void print();
    bool BackTrackingSolve(Location);
};

Board::Board() {}

Board::Board(array<array<int, 10>, 10> b) : board(b) {}

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
            cout << board[i][j] << ' ';
        cout << endl;
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

void Board::operator=(array<array<int, 10>, 10> b)
{
    board = b;
}
// Board B({{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0}, {0, 0, 0, 0, 0, 0, 0, 9, 0, 7}, {0, 5, 0, 0, 9, 7, 8, 1, 3, 0}, {0, 0, 0, 0, 0, 0, 2, 0, 8, 5}, {0, 9, 2, 0, 0, 1, 0, 8, 0, 3}, {0, 0, 0, 0, 3, 8, 5, 0, 0, 0}, {0, 6, 0, 8, 0, 4, 0, 0, 5, 1}, {0, 3, 4, 0, 6, 0, 0, 0, 0, 0}, {0, 0, 5, 2, 4, 9, 7, 0, 0, 6}, {0, 1, 0, 6, 0, 0, 0, 0, 0, 0}}});
Board B;

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

set<int> Location::row_choices() const
{
    set<int> choices = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 1; i <= 9; i++)
        choices.erase(B(x, i));
    return choices;
}
set<int> Location::column_choices() const
{
    set<int> choices = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 1; i <= 9; i++)
        choices.erase(B(i, y));
    return choices;
}
set<int> Location::grid_choices() const
{
    set<int> choices = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    for (int i = 1; i <= 3; i++)
        for (int j = 1; j <= 3; j++)
            choices.erase(B((x - 1) / 3 * 3 + i, (y - 1) / 3 * 3 + j));
    return choices;
}
set<int> Location::choices() const
{
    set<int> _choices, choices;
    set<int> RowChoices = row_choices(), ColumnChoices = column_choices(), GridChoices = grid_choices();
    set_intersection(RowChoices.begin(), RowChoices.end(), ColumnChoices.begin(), ColumnChoices.end(), inserter(_choices, _choices.begin()));
    set_intersection(_choices.begin(), _choices.end(), GridChoices.begin(), GridChoices.end(), inserter(choices, choices.begin()));
    if (x == y)
        for (int i = 1; i <= 9; i++)
            choices.erase(B(i, i));
    if (x + y == 10)
        for (int i = 1; i <= 9; i++)
            choices.erase(B(i, 10 - i));
    return choices;
}

int main()
{
    B = {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 9, 0, 7},
          {0, 5, 0, 0, 9, 7, 8, 1, 3, 0},
          {0, 0, 0, 0, 0, 0, 2, 0, 8, 5},
          {0, 9, 2, 0, 0, 1, 0, 8, 0, 3},
          {0, 0, 0, 0, 3, 8, 5, 0, 0, 0},
          {0, 6, 0, 8, 0, 4, 0, 0, 5, 1},
          {0, 3, 4, 0, 6, 0, 0, 0, 0, 0},
          {0, 0, 5, 2, 4, 9, 7, 0, 0, 6},
          {0, 1, 0, 6, 0, 0, 0, 0, 0, 0}}}; // 1
    // B = {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //       {0, 6, 7, 0, 0, 0, 0, 0, 0, 9},
    //       {0, 0, 9, 0, 6, 0, 0, 0, 0, 3},
    //       {0, 0, 3, 0, 0, 7, 0, 0, 0, 0},
    //       {0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    //       {0, 7, 0, 0, 0, 0, 0, 0, 0, 1},
    //       {0, 0, 0, 0, 0, 0, 0, 4, 0, 0},
    //       {0, 0, 0, 0, 0, 6, 0, 0, 2, 0},
    //       {0, 4, 0, 0, 0, 0, 3, 0, 8, 0},
    //       {0, 3, 0, 0, 0, 0, 0, 0, 5, 7}}}; // 3
    // B = {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //       {0, 6, 7, 0, 0, 0, 0, 0, 1, 9},
    //       {0, 8, 9, 0, 0, 0, 0, 0, 0, 3},
    //       {0, 1, 3, 2, 4, 7, 9, 5, 0, 8},
    //       {0, 9, 4, 1, 5, 3, 6, 8, 0, 2},
    //       {0, 7, 5, 3, 2, 4, 8, 6, 0, 1},
    //       {0, 2, 8, 6, 0, 9, 1, 4, 0, 5},
    //       {0, 5, 1, 8, 9, 6, 7, 3, 0, 4},
    //       {0, 4, 0, 7, 1, 5, 3, 9, 0, 6},
    //       {0, 3, 6, 9, 8, 2, 4, 1, 5, 7}}}; // test
    B.print();
    B.BackTrackingSolve(Location(1, 1));
    B.print();
    // Location l2(2, 8);
    // for (auto choice : l2.choices())
    //     cout << choice << ',';
    // cout << endl;
}
