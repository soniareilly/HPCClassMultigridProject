TAR1 = $(basename $(wildcard *.cpp))
TAR2 = $(wildcard *.o)

all: link1

link1: gs.cpp link2
	g++ -c $< -O3

link2: multigrid.cpp compile
	g++ -c $< -O3

compile: gs.o multigrid.o
	valgrind --leak-check=yes g++ $^ -o multigrid -O3 
# -fopenmp

clean:
	-$(RM) $(TAR1) $(TAR2) *~

.PHONY: all, clean