all: gs multigrid

gs: gs.cpp
	g++ gs.cpp -o gs -O3 -fopenmp

multigrid: multigrid.cpp
	g++ multigrid.cpp -o multigrid -O3 -fopenmp