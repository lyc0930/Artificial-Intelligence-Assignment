#include "header/AStar.hpp"
#include "header/Board.hpp"
#include "header/IDAStar.hpp"
#include "header/fileIO.hpp"
#include <ctime>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{
    // Node start({2, 3, 11, 4, 5, 0, 8, 14, 9, 10, 0, 7, 7, 12, 13, 1, 15, 7, 16, 18, 6, 19, 20, 17, 21}); // 1
    // Node start({1, 2, 3, 4, 5, 6, 7, 7, 8, 9, 14, 15, 7, 12, 10, 0, 0, 11, 17, 13, 19, 20, 16, 21, 18}); // 2
    // Node start({0, 6, 15, 7, 7, 8, 9, 13, 4, 7, 1, 2, 3, 10, 5, 14, 11, 16, 12, 18, 19, 20, 17, 21, 0}); // 3
    // Node start({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 0, 0, 19, 20, 21}); // test
    Node goal({1, 2, 3, 4, 5, 7, 7, 8, 9, 10, 6, 7, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 0, 0});
    if (argc != 3)
        exit(EXIT_FAILURE);
    string algorithm = argv[1];
    string number = argv[2];
    Node start(fin("../input/" + number + ".txt"));

    clock_t startTime = clock();
    if (algorithm == "AStar")
        cout << AStar::GraphSearch(start, goal).depth << endl;
    else if (algorithm == "IDAStar")
        cout << IDAStar::GraphSearch(start, goal).size() - 1 << endl;

    clock_t endTime = clock();
    cout << "运行时间：" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    if (algorithm == "AStar")
        fout("../output/" + number + ".txt", AStar::Movement(start.Position, goal.Position));
    else if (algorithm == "IDAStar")
        fout("../output/" + number + ".txt", IDAStar::Movement(start.Position, goal.Position));
    // cout << IDAStar::GraphSearch(start, goal).size() - 1 << endl;
    // cout << IDAStar::Movement(start.Position, goal.Position);
    return 0;
}