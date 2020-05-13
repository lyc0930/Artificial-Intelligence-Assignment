#include "AStar.hpp"
#include "Board.hpp"
#include "fileIO.hpp"
#include <ctime>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

int main()
{
    // Node start({2, 3, 11, 4, 5, 0, 8, 14, 9, 10, 0, 7, 7, 12, 13, 1, 15, 7, 16, 18, 6, 19, 20, 17, 21}); // 1
    // Node start({1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 14, 15, 7, 12, 10, 0, 0, 11, 17, 13, 19, 20, 16, 21, 18}); // 2
    // Node start({0, 6, 15, 7, 7, 8, 9, 13, 4, 7, 1, 2, 3, 10, 5, 14, 11, 16, 12, 18, 19, 20, 17, 21, 0}); // 3
    // Node start({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 0, 0, 19, 20, 21, 17, 18}); // test
    Node start(fin("..\\input\\1.txt"));
    Node goal({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0});

    clock_t startTime = clock();

    cout << AStar::GraphSearch(start, goal).depth << endl;
    clock_t endTime = clock();
    cout << "运行时间：" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    fout("..\\output\\1.txt", AStar::Movement(start.Position, goal.Position));
    return 0;
}