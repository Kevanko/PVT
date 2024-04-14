#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int THREADS = 2;

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double func(double x) { return (pow(x, 4) / ( 0.5* pow(x, 2) + x + 6)); }

void serial() {
  const double a = 0.4;
  const double b = 1.5;
  const int n = 10000;
  double h = (b - a) / n;
  double s = 0.0;
  double t = wtime();
  for (int i = 0; i < n; i++) s += func(a + h * (i + 0.5));
  s *= h;
  t = wtime() - t;
  printf("Elapsed time (serial): %.6f sec.\n", t);
  printf("Result: %.12f\n", s);
}

void parallel() {
  const double a = 0.4;
  const double b = 1.5;
  const int n = 10000;
  double h = (b - a) / n;
  double s = 0.0;
  double t = wtime();
#pragma omp parallel num_threads(THREADS)
  {
    double sloc = 0.0;
#pragma omp for nowait
    for (int i = 0; i < n; i++) sloc += func(a + h * (i + 0.5));
#pragma omp atomic
    s += sloc;
  }
  s *= h;
  t = wtime() - t;
  printf("Elapsed time (parallel): %.6f sec.\n", t);
  printf("Result: %.12f\n", s);
}

int main() {
  printf("VARIANT: %d\n", 11 % 6 + 1);
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