TAR1 = $(basename $(wildcard *.cpp))
TAR2 = $(wildcard *.o)

all: compile

compile: link1
	g++ -o multigrid gs.o multigrid.o -O0 -fopenmp -std=c++11 -lgomp

link1: link2
	g++ -fopenmp -std=c++11 -O0  -c -o gs.o gs.cpp

link2:
	g++ -fopenmp -std=c++11 -O0  -c -o multigrid.o multigrid.cpp

clean:
	-$(RM) $(TAR1) $(TAR2) *~

.PHONY: all, clean