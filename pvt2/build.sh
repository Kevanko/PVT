#!/bin/bash

gcc -o dgemv DGEMV.c -std=c99 -Wall -O2 -fopenmp

