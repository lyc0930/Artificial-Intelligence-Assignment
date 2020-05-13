#include "fileIO.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

void extend(const std::string str, std::vector<unsigned char> &board, const char delimeter)
{
    std::istringstream iss(str);
    std::string temp;

    while (std::getline(iss, temp, delimeter))
        board.emplace_back((unsigned char)(std::stoi(std::move(temp))));

    return;
}

std::vector<unsigned char> fin(std::string fileName)
{
    std::ifstream input(fileName, std::ios::in);
    std::string buffer;
    std::vector<unsigned char> board;
    if (!input.is_open())
    {
        std::cout << "Error opening file";
        exit(1);
    }
    while (!input.eof())
    {
        input >> buffer;
        extend(buffer, board, ',');
    }
    return board;
}

void fout(std::string fileName, std::string content)
{
    std::ofstream output(fileName, std::ios::out);
    if (!output.is_open())
    {
        std::cout << "Error opening file";
        exit(1);
    }
    output << content;
    return;
}
