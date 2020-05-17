all : fileIO.cpp fileIO.hpp AStar.cpp AStar.hpp IDAStar.cpp IDAStar.hpp Board.cpp Board.hpp digit.cpp
	g++ fileIO.cpp AStar.cpp IDAStar.cpp Board.cpp digit.cpp -O3 -o digit
fileIO.o : fileIO.cpp fileIO.hpp
	g++ -c fileIO.cpp
AStar.o : AStar.cpp AStar.hpp
	g++ -c AStar.cpp
IDAStar.o : IDAStar.cpp IDAStar.hpp
	g++ -c IDAStar.cpp
Board.o : Board.cpp Board.hpp
	g++ -c Board.cpp
digit.o : digit.cpp
	g++ -c digit.cpp
digit : digit.o Board.o AStar.o IDAStar.o fileIO.o
	g++ digit.o Board.o AStar.o IDAStar.o fileIO.o -O3 -o digit
clean :
	rm -f digit.o Board.o AStar.o fileIO.o