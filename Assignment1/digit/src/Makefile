fileIO.o : fileIO.cpp fileIO.hpp
	g++ -c fileIO.cpp
AStar.o : AStar.cpp AStar.hpp
	g++ -c AStar.cpp
Board.o : Board.cpp Board.hpp
	g++ -c Board.cpp
digit.o : digit.cpp
	g++ -c digit.cpp
digit : digit.o Board.o AStar.o fileIO.o
	g++ digit.o Board.o AStar.o fileIO.o -O3 -o digit
clean :
	rm -f digit.o Board.o AStar.o fileIO.o
all : fileIO.cpp fileIO.hpp AStar.cpp AStar.hpp Board.cpp Board.hpp digit.cpp
	g++ -c fileIO.cpp
	g++ -c AStar.cpp
	g++ -c Board.cpp
	g++ -c digit.cpp
	g++ digit.o Board.o AStar.o fileIO.o -O3 -o digit
