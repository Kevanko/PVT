#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int THREADS = 2;

/* pseudo-random number in the [0, 1] */
double getrand_serial() { return (double)rand() / RAND_MAX; }

double getrand_parallel(unsigned int *seed) {
  return (double)rand_r(seed) / RAND_MAX;
}

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double func(double x, double y) { return (exp(x - y)); }

const double PI = 3.14159265358979323846;
const int n = 10000000;

void serial() {
  int in = 0;
  double s = 0;
  double t = wtime();
  for (int i = 0; i < n; i++) {
    double x = getrand_serial() -1; /* x in [-1, 0] */
    double y = getrand_serial();      /* y in [0, 1] */
    if (y <= 1) {
      in++;
      s += func(x, y);
    }
  }
  double v = PI * in / n;
  double res = v * s / in;
  t = wtime() - t;
  printf("Elapsed time (serial): %.6f sec.\n", t);
  printf("Result: %.12f, n %d\n", res, n);
}

void parallel() {
  int in = 0;
  double s = 0;
  double t = wtime();
#pragma omp parallel
  {
    double s_loc = 0;
    int in_loc = 0;
    unsigned int seed = omp_get_thread_num();
#pragma omp for nowait
    for (int i = 0; i < n; i++) {
      double x = getrand_parallel(&seed) -1; /* x in [-1, 0] */
      double y = getrand_parallel(&seed);      /* y in [0, 1] */
      if (y <= 1) {
        in_loc++;
        s_loc += func(x, y);
      }
    }
#pragma omp atomic
    s += s_loc;
#pragma omp atomic
    in += in_loc;
  }
  double v = PI * in / n;
  double res = v * s / in;
  t = wtime() - t;
  printf("Elapsed time (serial): %.6f sec.\n", t);
  printf("Result: %.12f, n %d\n", res, n);
}

int main() {
      printf("VARIANT: %d\n", 11 % 3 + 1);
  for (; THREADS <= 8; THREADS += 2) {
    printf("-------------%d-------------\n", THREADS);
    printf("> serial\n");
    serial();
    printf("> parallel\n");
    parallel();
    printf("---------------------------\n");
  }
  return 0;
}