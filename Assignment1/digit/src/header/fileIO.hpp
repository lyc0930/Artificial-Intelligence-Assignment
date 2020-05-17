#ifndef FILEIO_HPP
#define FILEIO_HPP
#include "Board.hpp"
#include <fstream>
#include <string>
#include <vector>

void extend(const std::string str, std::vector<unsigned char> &board, const char delimeter);

std::vector<unsigned char> fin(std::string fileName);

void fout(std::string fileName, std::string content);

#endif // !FILEIO_HPP