# Hashlife Parallel

The code in this repository attempts to parallelize the Hashlife algorithm using OpenMP.

A modest speedup is achieved using 4 threads, although the parallelization fails to scale well beyond this point.
Also, the code seems to contain some bugs that result in rare segfaults with certain parameter combinations.
