all: multigrid gs

multigrid: multigrid.cpp
	g++ multigrid.cpp -o multigrid -O3 -fopenmp

gs: gs.cpp
	g++ gs.cpp -o gs -O3 -fopenmp