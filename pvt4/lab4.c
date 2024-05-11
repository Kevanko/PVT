#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <inttypes.h>
#include <stdlib.h>
#include <time.h>

double S(double time_nomp,double time_omp){
    return time_nomp/time_omp;
}

double wtime()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1E-9;
}

struct particle { float x, y, z; };

const float G = 6.67e-11;

omp_lock_t *locks; // Массив из N блокировок (мьютексов) — блокировка на уровне отдельных ячеек

void calculate_forces_serial(struct particle *p, struct particle *f, float *m, int n)
{
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            // Вычисление силы, действующей на тело i со стороны j
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            struct particle dir = {
                .x = p[j].x - p[i].x,
                .y = p[j].y - p[i].y,
                .z = p[j].z - p[i].z
            };//Расчёт вектора направления
            // Сумма сил, действующих на тело i
            f[i].x += mag * dir.x / dist;//dir.x/dist - расстояние
            f[i].y += mag * dir.y / dist;
            f[i].z += mag * dir.z / dist;
            // Сумма сил, действующих на тело j (симметричность)
            f[j].x -= mag * dir.x / dist;
            f[j].y -= mag * dir.y / dist;
            f[j].z -= mag * dir.z / dist;
        }
    }
}

void calculate_forces_one_critical(struct particle *p, struct particle *f, float *m, int n,int i)
{
    #pragma omp parallel num_threads(i)
    {
        #pragma omp parallel for
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                float dist = sqrtf(powf(p[i].x - p[j].x, 2) +
                powf(p[i].y - p[j].y, 2) +
                powf(p[i].z - p[j].z, 2));
                float mag = (G * m[i] * m[j]) / powf(dist, 2);
                struct particle dir = {
                    .x = p[j].x - p[i].x,
                    .y = p[j].y - p[i].y,
                    .z = p[j].z - p[i].z
                };
                #pragma omp critical
                {
                    f[i].x += mag * dir.x / dist;
                    f[i].y += mag * dir.y / dist;
                    f[i].z += mag * dir.z / dist;
                    f[j].x -= mag * dir.x / dist;
                    f[j].y -= mag * dir.y / dist;
                    f[j].z -= mag * dir.z / dist;
                }
            }
        }
    }
}//хуже чем послед

void calculate_forces_six_atomar(struct particle *p, struct particle *f, float *m, int n)
{
    #pragma omp for schedule(dynamic, 16) nowait // Циклическое распределение итераций
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) +
            powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            struct particle dir = {
                .x = p[j].x - p[i].x, .y = p[j].y - p[i].y, .z = p[j].z - p[i].z
            };
            #pragma omp atomic
                f[i].x += mag * dir.x / dist;
            #pragma omp atomic
                f[i].y += mag * dir.y / dist;
            #pragma omp atomic
                f[i].z += mag * dir.z / dist;
            #pragma omp atomic
                f[j].x -= mag * dir.x / dist;
            #pragma omp atomic
                f[j].y -= mag * dir.y / dist;
            #pragma omp atomic
                f[j].z -= mag * dir.z / dist;
        }
    }
}

void calculate_forces_nblock(struct particle *p, struct particle *f, float *m, int n)
{
    #pragma omp for schedule(dynamic, 16) nowait
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            struct particle dir = {
                .x = p[j].x - p[i].x,
                .y = p[j].y - p[i].y,
                .z = p[j].z - p[i].z
            };
            omp_set_lock(&locks[i]);
            f[i].x += mag * dir.x / dist;
            f[i].y += mag * dir.y / dist;
            f[i].z += mag * dir.z / dist;
            omp_unset_lock(&locks[i]);
            omp_set_lock(&locks[j]);
            f[j].x -= mag * dir.x / dist;
            f[j].y -= mag * dir.y / dist;
            f[j].z -= mag * dir.z / dist;
            omp_unset_lock(&locks[j]);
        }
    }
}

void calculate_forces_redundant_calculations(struct particle *p, struct particle *f, float *m, int n)
{
    #pragma omp for schedule(dynamic, 8) nowait
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j)
                continue;
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            struct particle dir = {
                .x = p[j].x - p[i].x,
                .y = p[j].y - p[i].y,
                .z = p[j].z - p[i].z
            };
            f[i].x += mag * dir.x / dist;
            f[i].y += mag * dir.y / dist;
            f[i].z += mag * dir.z / dist;
        }
    }
}//увеличели кол-во вычислений, но не будет гонки данных

void calculate_forces_stream_storage(struct particle *p, struct particle *f[], float *m, int n)
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for (int i = 0; i < n; i++) {
        f[tid][i].x = 0;
        f[tid][i].y = 0;
        f[tid][i].z = 0;
    }
    #pragma omp for schedule(dynamic, 8)
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) + powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            struct particle dir = { .x = p[j].x - p[i].x, .y = p[j].y - p[i].y, .z = p[j].z - p[i].z };
                f[tid][i].x += mag * dir.x / dist;
                f[tid][i].y += mag * dir.y / dist;
                f[tid][i].z += mag * dir.z / dist;
                f[tid][j].x -= mag * dir.x / dist;
                f[tid][j].y -= mag * dir.y / dist;
                f[tid][j].z -= mag * dir.z / dist;
        }
    } // barrier
    #pragma omp single // Итоговый вектор сил сформируем в первой строке – f[0][i]
    {
        for (int i = 0; i < n; i++) {
            for (int tid = 1; tid < nthreads; tid++) {
                f[0][i].x += f[tid][i].x;
                f[0][i].y += f[tid][i].y;
                f[0][i].z += f[tid][i].z;
            }
        }
    }
} //локальная копия вектора сил для каждого

void move_particles(struct particle *p, struct particle *f, struct particle *v, float *m, int n, double dt)
{
    for (int i = 0; i < n; i++) {
        struct particle dv = {
            .x = f[i].x / m[i] * dt,
            .y = f[i].y / m[i] * dt,
            .z = f[i].z / m[i] * dt,
        };
        struct particle dp = {
            .x = (v[i].x + dv.x / 2) * dt,
            .y = (v[i].y + dv.y / 2) * dt,
            .z = (v[i].z + dv.z / 2) * dt,
        };
        v[i].x += dv.x;
        v[i].y += dv.y;
        v[i].z += dv.z;
        p[i].x += dp.x;
        p[i].y += dp.y;
        p[i].z += dp.z;
        f[i].x = f[i].y = f[i].z = 0;
    }
}

void move_particles_nowait(struct particle *p, struct particle *f, struct particle *v,
float *m, int n, double dt)
{
    #pragma omp for nowait
    for (int i = 0; i < n; i++) {
        struct particle dv = {
            .x = f[i].x / m[i] * dt,
            .y = f[i].y / m[i] * dt,
            .z = f[i].z / m[i] * dt,
        };
        struct particle dp = {
            .x = (v[i].x + dv.x / 2) * dt,
            .y = (v[i].y + dv.y / 2) * dt,
            .z = (v[i].z + dv.z / 2) * dt,
        };
        v[i].x += dv.x;
        v[i].y += dv.y;
        v[i].z += dv.z;
        p[i].x += dp.x;
        p[i].y += dp.y;
        p[i].z += dp.z;
        f[i].x = f[i].y = f[i].z = 0;
    }
}

void move_particles_stream_storage(struct particle *p, struct particle *f[], struct particle *v, float *m, int n, double dt)
{
    #pragma omp for
    for (int i = 0; i < n; i++) {
        struct particle dv = {
            .x = f[0][i].x / m[i] * dt,
            .y = f[0][i].y / m[i] * dt,
            .z = f[0][i].z / m[i] * dt,
        };
        struct particle dp = {
            .x = (v[i].x + dv.x / 2) * dt,
            .y = (v[i].y + dv.y / 2) * dt,
            .z = (v[i].z + dv.z / 2) * dt,
        };
        v[i].x += dv.x;
        v[i].y += dv.y;
        v[i].z += dv.z;
        p[i].x += dp.x;
        p[i].y += dp.y;
        p[i].z += dp.z;
    }
}

double run_serial(){
    double ttotal, tinit = 0, tforces = 0, tmove = 0;
    ttotal = wtime();
    int n = 80;
    tinit = -wtime();
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f = malloc(sizeof(*f) * n); // Сила, действующая на каждую частицу (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n); // Масса частицы
    for (int i = 0; i < n; i++) {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        f[i].x = f[i].y = f[i].z = 0;
    }
    tinit += wtime();
    double dt = 1e-5;
    for (double t = 0; t <= 1; t += dt) { // Цикл по времени (модельному)
        tforces -= wtime();
        calculate_forces_serial(p, f, m, n); // Вычисление сил – O(N^2)
        tforces += wtime();
        tmove -= wtime();
        move_particles(p, f, v, m, n, dt); // Перемещение тел O(N)
        tmove += wtime();
    }
    ttotal = wtime() - ttotal;
    printf("# NBody (n=%d)\n", n);
    printf("# Elapsed time (sec): ttotal %.6f, tinit %.6f, tforces %.6f, tmove %.6f\n", ttotal, tinit, tforces, tmove);
    free(m);
    free(v);
    free(f);
    free(p);
    return ttotal;
}

double run_one_criticle(int i){
    double ttotal;
    ttotal = wtime();
    int n = 80;
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f = malloc(sizeof(*f) * n); // Сила, действующая на каждую частицу (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n); // Масса частицы
    for (int i = 0; i < n; i++) {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        f[i].x = f[i].y = f[i].z = 0;
    }
    double dt = 1e-5;
    for (double t = 0; t <= 1; t += dt) { // Цикл по времени (модельному)
        calculate_forces_one_critical(p, f, m, n, i); // Вычисление сил – O(N^2)
        move_particles(p, f, v, m, n, dt); // Перемещение тел O(N)
    }
    ttotal = wtime() - ttotal;
    printf("# NBody (n=%d)\n", n);
    printf("# Elapsed time (sec): ttotal %.6f\n", ttotal);
    free(m);
    free(v);
    free(f);
    free(p);
    return ttotal;
}

double run_six_atomar(int i){
    double ttotal;
    ttotal = wtime();
    int n = 80;
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f = malloc(sizeof(*f) * n); // Сила, действующая на каждую частицу (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n); // Масса частицы
    for (int i = 0; i < n; i++) {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        f[i].x = f[i].y = f[i].z = 0;
    }
    double dt = 1e-5;
    #pragma omp parallel num_threads(i)// Параллельный регион активируется один раз
    {
        for (double t = 0; t <= 1; t += dt) {
            calculate_forces_six_atomar(p, f, m, n);
            #pragma omp barrier // Ожидание завершения расчетов f[i]
            move_particles_nowait(p, f, v, m, n, dt);
            #pragma omp barrier // Ожидание завершения обновления p[i], f[i]
        }
    }
    ttotal = wtime() - ttotal;
    printf("# NBody (n=%d)\n", n);
    printf("# Elapsed time (sec): ttotal %.6f\n", ttotal);
    free(m);
    free(v);
    free(f);
    free(p);
    return ttotal;
}

double run_nblock(int i){
    double ttotal;
    ttotal = wtime();
    int n = 80;
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f = malloc(sizeof(*f) * n); // Сила, действующая на каждую частицу (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n); // Масса частицы
    for (int i = 0; i < n; i++) {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        f[i].x = f[i].y = f[i].z = 0;
    }
    locks = malloc(sizeof(omp_lock_t) * n);
    for (int i = 0; i < n; i++)
        omp_init_lock(&locks[i]);
    double dt = 1e-5;
    #pragma omp parallel num_threads(i)
    {
        for (double t = 0; t <= 1; t += dt) {
            calculate_forces_nblock(p, f, m, n);
            #pragma omp barrier
            move_particles_nowait(p, f, v, m, n, dt);
            #pragma omp barrier
        }
    }
    ttotal = wtime() - ttotal;
    printf("# NBody (n=%d)\n", n);
    printf("# Elapsed time (sec): ttotal %.6f\n",ttotal);
    free(m);
    free(v);
    free(f);
    free(p);
    free(locks);
    return ttotal;
}

double run_redundant_calculations(int i){
    double ttotal;
    ttotal = wtime();
    int n = 80;
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f = malloc(sizeof(*f) * n); // Сила, действующая на каждую частицу (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n); // Масса частицы
    for (int i = 0; i < n; i++) {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        f[i].x = f[i].y = f[i].z = 0;
    }
    double dt = 1e-5;
    #pragma omp parallel num_threads(i)
    {
        for (double t = 0; t <= 1; t += dt) { // Цикл по времени (модельному)
            calculate_forces_redundant_calculations(p, f, m, n); // Вычисление сил – O(N^2)
            #pragma omp barrier
            move_particles_nowait(p, f, v, m, n, dt); // Перемещение тел O(N)
            #pragma omp barrier
        }
    }
    ttotal = wtime() - ttotal;
    printf("# NBody (n=%d)\n", n);
    printf("# Elapsed time (sec): ttotal %.6f\n",ttotal);
    free(m);
    free(v);
    free(f);
    free(p);
    return ttotal;
}

double run_stream_storage(int i){
    double ttotal;
    ttotal = wtime();
    int n = 80;
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); i++)
        f[i] = malloc(sizeof(struct particle) * n);
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n); // Масса частицы
    for (int i = 0; i < n; i++) {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
    }
    double dt = 1e-5;
    #pragma omp parallel num_threads(i)
    {
        for (double t = 0; t <= 1; t += dt) { // Цикл по времени (модельному)
            calculate_forces_stream_storage(p, f, m, n); // Вычисление сил – O(N^2)
            move_particles_stream_storage(p, f, v, m, n, dt); // Перемещение тел O(N)
        }
    }
    ttotal = wtime() - ttotal;
    printf("# NBody (n=%d)\n", n);
    printf("# Elapsed time (sec): ttotal %.6f\n",ttotal);
    free(m);
    free(v);
    free(p);
    return ttotal;
}

void write_one_criticle(double S, int n){ 
  FILE *f; 
  f = fopen("res1.txt", "a"); 
  fprintf(f, "%d %f\n", n, S); 
  fclose(f); 
}

void write_six_atomar(double S, int n){ 
  FILE *f; 
  f = fopen("res2.txt", "a"); 
  fprintf(f, "%d %f\n", n, S); 
  fclose(f); 
} 

void write_nblock(double S, int n){ 
  FILE *f; 
  f = fopen("res3.txt", "a"); 
  fprintf(f, "%d %f\n", n, S); 
  fclose(f); 
} 

void write_redundant_calculations(double S, int n){ 
  FILE *f; 
  f = fopen("res4.txt", "a"); 
  fprintf(f, "%d %f\n", n, S); 
  fclose(f); 
} 

void write_stream_storage(double S, int n){ 
  FILE *f; 
  f = fopen("res5.txt", "a"); 
  fprintf(f, "%d %f\n", n, S); 
  fclose(f); 
} 

int main()
{
    double TtotalSerial = run_serial();
    for(int i = 2; i<5; i++){
        double TtotalOneCriticle = run_one_criticle(i);
        double TtotalSixAtomar = run_six_atomar(i);
        double TtotalNBlock = run_nblock(i);
        double TtotalRedundantCalculations = run_redundant_calculations(i);
        double TtotalStreamStorage = run_stream_storage(i);
        double ResOneCriticle = S(TtotalSerial,TtotalOneCriticle);
        double ResSixAtomar = S(TtotalSerial,TtotalSixAtomar);
        double ResNBlock = S(TtotalSerial,TtotalNBlock);
        double ResRedundantCalculations = S(TtotalSerial,TtotalRedundantCalculations);
        double ResStreamStorage = S(TtotalSerial,TtotalStreamStorage);
        printf("One critical result: %f\n",ResOneCriticle);
        printf("Six atomar result: %f\n",ResSixAtomar);
        printf("N Block result: %f\n",ResNBlock);
        printf("Redundant calculations result: %f\n",ResRedundantCalculations);
        printf("Stream storage result: %f\n",ResStreamStorage);
        write_one_criticle(ResOneCriticle,i);
        write_six_atomar(ResSixAtomar,i);
        write_nblock(ResNBlock,i);
        write_redundant_calculations(ResRedundantCalculations,i);
        write_stream_storage(ResStreamStorage,i);
    }
    return 0;
}