#ifndef BOARD_HPP
#define BOARD_HPP
#include <array>
#include <set>

class Location
{
public:
    int x, y;
    Location(int, int);
    Location(const Location &);
    Location next() const;
    Location &operator++();
    Location operator++(int);
    Location &operator=(const Location &);
    explicit operator bool() const;
    std::set<int> row_choices() const;
    std::set<int> column_choices() const;
    std::set<int> grid_choices() const;
    std::set<int> choices() const;
};

class Board
{
public:
    std::array<std::array<int, 10>, 10> board;
    Board();
    Board(std::array<std::array<int, 10>, 10>);
    int &operator()(int, int);
    int &operator[](Location);
    void operator=(std::array<std::array<int, 10>, 10>);
    void print();
    bool BackTrackingSolve(Location);
};

extern Board B;

#endif // !BOARD_HPP