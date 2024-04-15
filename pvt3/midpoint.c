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
double eps = 1E-5;
const int n0 = 1000000;

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

double func(double x) { return (pow(x, 4) / (0.5 * pow(x, 2) + x + 6)); }

void serial() {
  const double a = 0.4;
  const double b = 1.5;
  int n = n0, k;
  double sq[2], delta = 1;
  t_serial = wtime();
  for (k = 0; delta > eps; n *= 2, k ^= 1) {
    double h = (b - a) / n;
    double s = 0.0;
    for (int i = 0; i < n; i++) s += func(a + h * (i + 0.5));
    sq[k] = s * h;
    if (n > n0) delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
  }
  t_serial = wtime() - t_serial;
  printf("Elapsed time (serial): %.6f sec.\n", t_serial);
  printf("Result: %.12f\n", sq[k]);
}

void parallel() {
  const double a = 0.4;
  const double b = 1.5;
  double sq[2];
  t_parallel = wtime();
#pragma omp parallel num_threads(THREADS)
  {
    int n = n0, k;
    double delta = 1;
    for (k = 0; delta > eps; n *= 2, k ^= 1) {
      double h = (b - a) / n;
      double s = 0.0;
      sq[k] = 0;

// Ждем пока все потоки закончат обнуление sq[k], s
#pragma omp barrier

#pragma omp for nowait
      for (int i = 0; i < n; i++) {
        s += func(a + h * (i + 0.5));
      }

#pragma omp atomic
      sq[k] += s * h;
// Ждем пока все потоки обновят sq[k]
#pragma omp barrier
      if (n > n0) delta = fabs(sq[k] - sq[k ^ 1]) / 3.0;
#if 0
        printf("n=%d i=%d sq=%.12f delta=%.12f\n", n, k, sq[k], delta);
#endif
    }
  }
  t_parallel = wtime() - t_parallel;
  printf("Elapsed time (parallel): %.6f sec.\n", t_parallel);
  printf("Result: %.12f\n", sq[1]);
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