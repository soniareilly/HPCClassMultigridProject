TAR1 = $(basename $(wildcard *.cpp))
TAR2 = $(wildcard *.o)
CU1 = $(basename $(wildcard *.cu))

all: compile cuda

compile: link1
	g++ -o multigrid gs.o multigrid.o -O0 -fopenmp -std=c++11 -lgomp

link1: link2
	g++ -fopenmp -std=c++11 -O0  -c -o gs.o gs.cpp

link2:
	g++ -fopenmp -std=c++11 -O0  -c -o multigrid.o multigrid.cpp

cuda: cuda1
	nvcc -o multigrid_cu -O0 gscu.o multigrid_cu.o

cuda1: cuda2
	nvcc -c -o multigrid_cu.o -O0 multigrid.cu

cuda2: 
	nvcc -c -o gscu.o -O0 gs.cu

clean:
	-$(RM) $(TAR1) $(TAR2) $(CU1) *~

.PHONY: all, clean