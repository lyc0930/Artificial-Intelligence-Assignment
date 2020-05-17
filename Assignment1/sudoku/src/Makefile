all : fileIO.cpp fileIO.hpp Board.cpp Board.hpp sudoku.cpp
	g++ fileIO.cpp Board.cpp sudoku.cpp -O3 -o sudoku
fileIO.o : fileIO.cpp fileIO.hpp
	g++ -c fileIO.cpp
Board.o : Board.cpp Board.hpp
	g++ -c Board.cpp
sudoku.o : sudoku.cpp
	g++ -c sudoku.cpp
sudoku : sudoku.o Board.o fileIO.o
	g++ sudoku.o Board.o fileIO.o -O3 -o sudoku
clean :
	rm -f sudoku.o Board.o fileIO.o