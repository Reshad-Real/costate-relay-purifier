/* PC benchmark: reads instances written by experiments/e6_timing.py, solves each in three modes,
 * prints CSV: id,mode,J,nodes,leaves,time_us,seq0  (time = median of `reps` repetitions). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpc_solver.h"
#ifdef _WIN32
#include <windows.h>
static double now_us(void) {
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e6 / (double)f.QuadPart;
}
#else
#include <time.h>
static double now_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e6 + t.tv_nsec / 1e3; }
#endif

static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }

static int rd_i(FILE *f) { int v; if (fread(&v, 4, 1, f) != 1) exit(2); return v; }
static float *rd_f(FILE *f, int n) { float *p = (float *)malloc(sizeof(float) * (n > 0 ? n : 1)); if (n && fread(p, 4, n, f) != (size_t)n) exit(3); return p; }

int main(int argc, char **argv)
{
    FILE *f = fopen(argv[1], "rb");
    int reps = argc > 2 ? atoi(argv[2]) : 5, ninst, id, i, m;
    static mpc_work_t w;
    if (!f) { fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    ninst = rd_i(f);
    printf("id,mode,J,nodes,leaves,time_us,seq0\n");
    for (id = 0; id < ninst; ++id) {
        mpc_inst_t in;
        float *hdr, *A, *G, *W, *Wf, *cb, *D, *occ, *c0;
        int has_wf, n, sz;
        in.nz = rd_i(f); in.np = rd_i(f); in.nu = rd_i(f); in.N = rd_i(f); in.Nu = rd_i(f);
        for (i = 0; i < in.Nu; ++i) in.blen[i] = rd_i(f);
        has_wf = rd_i(f);
        hdr = rd_f(f, 4); in.dt = hdr[0]; in.rho = hdr[1]; in.M = hdr[2]; in.e_allow = hdr[3];
        { float *e = rd_f(f, in.nu); for (i = 0; i < in.nu; ++i) in.e_step[i] = e[i]; free(e); }
        n = in.np * in.nz; sz = in.nu * in.np * in.nz * in.nz;
        A = rd_f(f, sz); G = rd_f(f, sz); W = rd_f(f, n); Wf = rd_f(f, n); cb = rd_f(f, n);
        D = rd_f(f, in.N * n); occ = rd_f(f, in.N); c0 = rd_f(f, n);
        in.A = A; in.G = G; in.W = W; in.Wf = has_wf ? Wf : NULL; in.cb = cb; in.D = D; in.occ = occ; in.c0 = c0;
        for (m = 0; m < 3; ++m) {
            double ts[64]; int seq[MPC_MAXNU], r; float J = 0;
            int R = reps < 64 ? reps : 64;
            for (r = 0; r < R; ++r) {
                double t0 = now_us();
                mpc_solve(&in, m, &w, seq, &J);
                ts[r] = now_us() - t0;
            }
            qsort(ts, R, sizeof(double), cmpd);
            printf("%d,%d,%.9g,%ld,%ld,%.3f,%d\n", id, m, J, w.nodes, w.leaves, ts[R / 2], seq[0]);
        }
        free(hdr); free(A); free(G); free(W); free(Wf); free(cb); free(D); free(occ); free(c0);
    }
    fclose(f);
    return 0;
}
