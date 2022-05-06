# HPCClassMultigridProject
Final Project for HPC Spring 2022

## Description 

Advection-dffusion

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial u}{\partial t} + \mathbf{v}\cdot \nabla u + \nu\nabla^2 u= 0">
<img src="https://render.githubusercontent.com/render/math?math=u|_{\partial\Omega} = 0">
<img src="https://render.githubusercontent.com/render/math?math=u(x,y,0) = u_0(x,y)">

## Instruction

### Required packages/libararies

1. `CUDA` (version 9 or higher)
2. `OpenMP` (with `-fopenmp` compiler flag)

### `make`

Compile files with `.cpp` (serial and OMP version) and `.cu` extensions (CUDA version).\

### `.\multigrid`

Runs the serial multigrid as reference (with input parameters).\
Also runs the OMP version with specified number of threads.\
Compares the difference and runtime between the serial version and the OMP version.

### `.\multigrid_cu`

Runs the CUDA version of multigrid on GPU (if available).\
Initiates the timer for CUDA and compares with the referenced serial code.

### `python uTplot.py`

Plots the numerical solutions at 2D grid points at final time (colormap).

### `python strongsc_plot.py`

Plots the strong scaling visualization for OMP outputs (time vs. number of threads).

### `python uTerr.py`

Plots the performance (speed-up) for serial, OMP, and CUDA solutions (time vs. spatial dimension N).

