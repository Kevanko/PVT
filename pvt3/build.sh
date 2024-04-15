#!/bin/bash

gcc -o mid midpoint.c -fopenmp -lm
gcc -o carlo monte-carlo.c -fopenmp -lm

./mid 
./carlo
