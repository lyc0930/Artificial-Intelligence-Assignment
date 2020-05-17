all : source/fileIO.cpp header/fileIO.hpp source/AStar.cpp header/AStar.hpp source/IDAStar.cpp header/IDAStar.hpp source/Board.cpp header/Board.hpp digit.cpp
	g++ source/fileIO.cpp source/AStar.cpp source/IDAStar.cpp source/Board.cpp digit.cpp -O3 -o digit
fileIO.o : source/fileIO.cpp header/fileIO.hpp
	g++ -c source/fileIO.cpp
AStar.o : source/AStar.cpp header/AStar.hpp
	g++ -c source/AStar.cpp
IDAStar.o : source/IDAStar.cpp header/IDAStar.hpp
	g++ -c source/IDAStar.cpp
Board.o : source/Board.cpp header/Board.hpp
	g++ -c source/Board.cpp
digit.o : digit.cpp
	g++ -c digit.cpp
digit : digit.o Board.o AStar.o IDAStar.o fileIO.o
	g++ digit.o Board.o AStar.o IDAStar.o fileIO.o -O3 -o digit
clean :
	rm -f digit.o Board.o AStar.o IDAStar.o fileIO.o