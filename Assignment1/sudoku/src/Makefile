all : source/fileIO.cpp header/fileIO.hpp source/Board.cpp header/Board.hpp sudoku.cpp
	g++ source/fileIO.cpp source/Board.cpp sudoku.cpp -O3 -o sudoku
fileIO.o : source/fileIO.cpp header/fileIO.hpp
	g++ -c source/fileIO.cpp
Board.o : source/Board.cpp header/Board.hpp
	g++ -c source/Board.cpp
sudoku.o : sudoku.cpp
	g++ -c sudoku.cpp
sudoku : sudoku.o Board.o fileIO.o
	g++ sudoku.o Board.o fileIO.o -O3 -o sudoku
clean :
	rm -f sudoku.o Board.o fileIO.o