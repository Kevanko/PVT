#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

double S(double time_nomp,double time_omp){
    return time_nomp/time_omp;
}

double wtime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1E-9;
}

void partition(int *v, int *i, int *j, int low, int high) {
    *i = low;
    *j = high;
    int pivot = v[(low + high) / 2];
    do {
        while (v[*i] < pivot) (*i)++;
        while (v[*j] > pivot) (*j)--;
        if (*i <= *j) {
            int temp = v[*i];
            v[*i] = v[*j];
            v[*j] = temp;
            (*i)++;
            (*j)--;
        }
    } while (*i <= *j);
}

void quicksort(int *v, int low, int high) {
    int i, j;
    partition(v, &i, &j, low, high);
    if (low < j)
        quicksort(v, low, j);
    if (i < high)
        quicksort(v, i, high);
}

void quicksort_tasks(int *v, int low, int high) {
    int i, j;
    int threshold = 1000;
    partition(v, &i, &j, low, high);
    if (high - low < threshold || (j - low < threshold || high - i < threshold)) {
        if (low < j)
            quicksort_tasks(v, low, j);
        if(i < high)
            quicksort_tasks(v, i, high);
    } else {
        #pragma omp task untied
        { quicksort_tasks(v, low, j); }
        quicksort_tasks(v, i, high);
    }
}

void write(double S, int n){ 
  FILE *f; 
  f = fopen("res.txt", "a"); 
  fprintf(f, "%d %f\n", n, S); 
  fclose(f); 
} 

void rand_arr(int* arr, int size){
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 10000 + 1;
    }
}

int main() {
    const int size = 2000000;
    int arr[size];
    rand_arr(arr, size);
    // Сортировка массива
    double t = wtime();
    quicksort(arr, 0, size - 1);
    t = wtime() - t;
    printf("%f \n",t);
    for(int f = 2; f<=8; f+=2){
        rand_arr(arr, size);
        double time_omp=0;
        time_omp = wtime();
        #pragma omp parallel num_threads(f)
        {
            #pragma omp single
            quicksort_tasks(arr, 0, size - 1);
        }
        time_omp = wtime() - time_omp;
        printf("S: %.6f\n\n",S(t,time_omp));
        write(S(t,time_omp),f);
    }
    return 0;
}