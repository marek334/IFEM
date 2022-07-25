# IFEM

Running the calculation:

1) At first one has to generate horizontal position of nodes on the
Earth’s surface using Matlab code: octahedronQuadSphere_duplicates_m.m

2) In case of computations on the real Earth’s surface one has to
generate and add ellipsoidal heights, otherwise H=0 can be used.

3) For these nodes one has to generate boundary condition - the surface
gravity disturbances using, e.g. "GrafLab":
https://github.com/blazej-bucha/graflab-cookbook

4) Examples of input files to main program can be found in folder data Then one
has to run IFEM_parallel.c

5) To visualize results, one can use Plot2Dfast_QUAD_robinson.m
