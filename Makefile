all: multigrid gs

multigrid: multigrid.cpp
	g++ multigrid.cpp -o multigrid -O3 -Xcompiler -fopenmp

gs: gs.cpp
	g++ gs.cpp -o gs -O3 -Xcompiler -fopenmp