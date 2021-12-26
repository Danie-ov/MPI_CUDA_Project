Daniel Niazov 207437997

Divide the processes to jobs:
all even processes doing the calculation with openMP
all odd processes doing the clculation with cuda

pack the arrays of scores/offsets/mutants/weights and all the variables and send them to their processes