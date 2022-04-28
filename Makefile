TAR1 = $(basename $(wildcard *.cpp))
TAR2 = $(wildcard *.o)

all: link1

link1: gs.cpp link2
	g++ -c $< -O0 -fopenmp

link2: multigrid.cpp compile
	g++ -c $< -O0 -fopenmp

compile: gs.o multigrid.o
	g++ $^ -o multigrid -O0 -g -fopenmp
# -fopenmp

clean:
	-$(RM) $(TAR1) $(TAR2) *~

.PHONY: all, clean