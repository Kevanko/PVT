#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

int THREADS = 2;

double t_serial = 0.0;
double t_parallel = 0.0;

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double func(double x) { return (pow(x, 4) / (0.5 * pow(x, 2) + x + 6)); }

void serial() {
  const double a = 0.4;
  const double b = 1.5;
  const int n = 10000;
  double h = (b - a) / n;
  double s = 0.0;
  t_serial = wtime();
  for (int i = 0; i < n; i++) s += func(a + h * (i + 0.5));
  s *= h;
  t_serial = wtime() - t_serial;
  printf("Elapsed time (serial): %.6f sec.\n", t_serial);
  printf("Result: %.12f\n", s);
}

void parallel() {
  const double a = 0.4;
  const double b = 1.5;
  const int n = 10000;
  double h = (b - a) / n;
  double s = 0.0;
  t_parallel = wtime();
#pragma omp parallel num_threads(THREADS)
  {
    double sloc = 0.0;
#pragma omp for nowait
    for (int i = 0; i < n; i++) sloc += func(a + h * (i + 0.5));
#pragma omp atomic
    s += sloc;
  }
  s *= h;
  t_parallel = wtime() - t_parallel;
  printf("Elapsed time (parallel): %.6f sec.\n", t_parallel);
  printf("Result: %.12f\n", s);
}

int main() {
  printf("VARIANT: %d\n", 11 % 6 + 1);
  char buff[100] = "# Threads   Speedup\n";
  for (; THREADS <= 8; THREADS += 2) {
    printf("-------------%d-------------\n", THREADS);
    printf("> serial\n");
    serial();
    printf("> parallel\n");
    parallel();
    printf("> speed\n");
    printf("s: %f\n", t_serial / t_parallel);
    printf("---------------------------\n");
    char tmp[20];
    sprintf(tmp, "%d\t\t%f\n", THREADS, t_serial / t_parallel);
    strcat(buff, tmp);
  }
  strcat(buff, "\0");
  FILE* file = fopen("prog-midpoint.dat", "w");
  fwrite(buff, sizeof(char), strlen(buff), file);
  fclose(file);
  return 0;
}