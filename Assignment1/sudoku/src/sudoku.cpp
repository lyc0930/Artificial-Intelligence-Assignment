#include "header/Board.hpp"
#include "header/fileIO.hpp"
#include <ctime>
#include <iostream>
using namespace std;

int main(int argc, char *argv[])
{
    // B = {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //       {0, 0, 0, 0, 0, 0, 0, 9, 0, 7},
    //       {0, 5, 0, 0, 9, 7, 8, 1, 3, 0},
    //       {0, 0, 0, 0, 0, 0, 2, 0, 8, 5},
    //       {0, 9, 2, 0, 0, 1, 0, 8, 0, 3},
    //       {0, 0, 0, 0, 3, 8, 5, 0, 0, 0},
    //       {0, 6, 0, 8, 0, 4, 0, 0, 5, 1},
    //       {0, 3, 4, 0, 6, 0, 0, 0, 0, 0},
    //       {0, 0, 5, 2, 4, 9, 7, 0, 0, 6},
    //       {0, 1, 0, 6, 0, 0, 0, 0, 0, 0}}}; // 1
    // B = {{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //       {0, 0, 0, 5, 3, 0, 0, 0, 0, 2},
    //       {0, 0, 3, 0, 0, 0, 0, 4, 0, 9},
    //       {0, 0, 7, 0, 0, 2, 0, 0, 0, 0},
    //       {0, 2, 0, 7, 0, 0, 0, 0, 0, 1},
    //       {0, 3, 6, 4, 0, 0, 0, 9, 8, 5},
    //       {0, 5, 0, 0, 0, 0, 0, 7, 0, 6},
    //       {0, 0, 0, 0, 0, 4, 0, 0, 9, 0},
    //       {0, 7, 0, 1, 0, 0, 0, 0, 5, 0},
    //       {0, 4, 0, 0, 0, 0, 8, 1, 0, 0}}}; // 2
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

    // B.print();
    // B.BackTrackingSolve(Location(1, 1));
    // B.print();

    if (argc != 2)
        exit(EXIT_FAILURE);
    string number = argv[1];
    B = fin("../input/sudoku0" + number + ".txt");

    clock_t startTime = clock();
    B.BackTrackingSolve(Location(1, 1));
    clock_t endTime = clock();
    cout << "运行时间：" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    fout("../output/sudoku0" + number + ".txt", B.board);
}