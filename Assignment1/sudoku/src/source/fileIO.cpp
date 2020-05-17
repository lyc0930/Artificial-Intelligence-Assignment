#include "../header/fileIO.hpp"
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

std::array<std::array<int, 10>, 10> fin(std::string fileName)
{
    std::ifstream input(fileName, std::ios::in);
    std::string line;
    std::array<std::array<int, 10>, 10> content;
    if (!input.is_open())
    {
        std::cout << "Error opening file";
        exit(1);
    }
    for (int i = 1; i <= 9; i++)
    {
        std::getline(input, line);
        std::istringstream iss(line);
        int n;
        for (int j = 1; j <= 9; j++)
        {
            iss >> n;
            content[i][j] = n;
        }
    }
    return content;
}

void fout(std::string fileName, std::array<std::array<int, 10>, 10> board)
{
    std::ofstream output(fileName, std::ios::out);
    if (!output.is_open())
    {
        std::cout << "Error opening file";
        exit(1);
    }
    for (int i = 1; i <= 9; i++)
    {
        for (int j = 1; j <= 9; j++)
            output << board[i][j] << ' ';
        output << std::endl;
    }
    return;
}