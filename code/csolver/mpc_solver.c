/* See mpc_solver.h. Portable C99, single precision, no dynamic allocation. */
#include "mpc_solver.h"
#include <float.h>
#include <string.h>

/* Optional floating-point operation counter (build with -DMPC_COUNT); no effect otherwise. */
#ifdef MPC_COUNT
unsigned long long mpc_flops = 0;
#define FL(n) (mpc_flops += (unsigned long long)(n))
#else
#define FL(n) ((void)0)
#endif

#define IDX_A(in, u, p, i, j) ((((u) * (in)->np + (p)) * (in)->nz + (i)) * (in)->nz + (j))

static float stage_cost(const mpc_inst_t *in, const float *x, int k)
{
    int n = in->np * in->nz, i;
    float idx = 0.0f, pen = 0.0f;
    FL(3 * n + 4);
    for (i = 0; i < n; ++i) {
        idx += in->W[i] * x[i];
        if (in->M > 0.0f && x[i] > in->cb[i]) { pen += (x[i] - in->cb[i]) / in->cb[i]; FL(3); }
    }
    return in->dt * (in->occ[k] * idx + in->M * pen);
}

static void propagate(const mpc_inst_t *in, int u, const float *x, const float *gd, float *y)
{
    int p, i, j, nz = in->nz;
    FL(2 * in->np * nz * nz);
    for (p = 0; p < in->np; ++p) {
        const float *A = in->A + IDX_A(in, u, p, 0, 0);
        for (i = 0; i < nz; ++i) {
            float s = gd[p * nz + i];
            for (j = 0; j < nz; ++j) s += A[i * nz + j] * x[p * nz + j];
            y[p * nz + i] = s;
        }
    }
}

static void propagate_lb(const mpc_inst_t *in, mpc_work_t *w, const float *x, const float *gd, float *y)
{
    int p, i, j, nz = in->nz;
    FL(2 * in->np * nz * nz);
    for (p = 0; p < in->np; ++p)
        for (i = 0; i < nz; ++i) {
            float s = gd[p * nz + i];
            for (j = 0; j < nz; ++j) s += w->Alb[p][i][j] * x[p * nz + j];
            y[p * nz + i] = s;
        }
}

static float terminal(const mpc_inst_t *in, const float *x)
{
    int n = in->np * in->nz, i;
    float s = 0.0f;
    if (!in->Wf) return 0.0f;
    FL(2 * n);
    for (i = 0; i < n; ++i) s += in->Wf[i] * x[i];
    return s;
}

static float bound_forward(const mpc_inst_t *in, mpc_work_t *w, const float *x0, int k0)
{
    float xa[MPC_MAXP * MPC_MAXZ], xb[MPC_MAXP * MPC_MAXZ];
    float *x = xa, *y = xb, *t, lb = 0.0f;
    int k, n = in->np * in->nz;
    memcpy(x, x0, sizeof(float) * n);
    for (k = k0; k < in->N; ++k) {
        propagate_lb(in, w, x, w->GlbD[k], y);
        lb += stage_cost(in, y, k);
        t = x; x = y; y = t;
    }
    return lb + terminal(in, x);
}

static float bound_affine(const mpc_inst_t *in, mpc_work_t *w, const float *x, int k)
{
    int n = in->np * in->nz, i;
    float s = w->beta[k];
    FL(2 * n);
    for (i = 0; i < n; ++i) s += w->lam[k][i] * x[i];
    return s;
}

static void prepare(const mpc_inst_t *in, mpc_work_t *w)
{
    int u, k, p, i, j, nz = in->nz, np = in->np, n = np * nz;
    float Glb[MPC_MAXP][MPC_MAXZ][MPC_MAXZ];
    FL(in->N * np * nz * nz * 2 * (in->nu + 1) + in->N * (3 * n + 2 * np * nz * nz + 2 * n));
    for (p = 0; p < np; ++p)
        for (i = 0; i < nz; ++i)
            for (j = 0; j < nz; ++j) {
                float a = FLT_MAX, g = FLT_MAX;
                for (u = 0; u < in->nu; ++u) {
                    float av = in->A[IDX_A(in, u, p, i, j)], gv = in->G[IDX_A(in, u, p, i, j)];
                    if (av < a) a = av;
                    if (gv < g) g = gv;
                }
                w->Alb[p][i][j] = a;
                Glb[p][i][j] = g;
            }
    for (k = 0; k < in->N; ++k) {
        const float *d = in->D + k * n;
        for (p = 0; p < np; ++p)
            for (i = 0; i < nz; ++i) {
                float sl = 0.0f;
                for (j = 0; j < nz; ++j) sl += Glb[p][i][j] * d[p * nz + j];
                w->GlbD[k][p * nz + i] = sl;
                for (u = 0; u < in->nu; ++u) {
                    float s = 0.0f;
                    for (j = 0; j < nz; ++j) s += in->G[IDX_A(in, u, p, i, j)] * d[p * nz + j];
                    w->GD[u][k][p * nz + i] = s;
                }
            }
    }
    /* affine cost-to-go of the comparison system (penalty omitted): V_k(x) = lam_k.x + beta_k */
    for (i = 0; i < n; ++i) w->lam[in->N][i] = in->Wf ? in->Wf[i] : 0.0f;
    w->beta[in->N] = 0.0f;
    for (k = in->N - 1; k >= 0; --k) {
        float v[MPC_MAXP * MPC_MAXZ], b = w->beta[k + 1];
        for (i = 0; i < n; ++i) v[i] = in->occ[k] * in->dt * in->W[i] + w->lam[k + 1][i];
        for (p = 0; p < np; ++p)
            for (j = 0; j < nz; ++j) {
                float s = 0.0f;
                for (i = 0; i < nz; ++i) s += w->Alb[p][i][j] * v[p * nz + i];
                w->lam[k][p * nz + j] = s;
            }
        for (i = 0; i < n; ++i) b += v[i] * w->GlbD[k][i];
        w->beta[k] = b;
    }
    w->start[0] = 0;
    for (i = 0; i < in->Nu; ++i) w->start[i + 1] = w->start[i] + in->blen[i];
}

static float eval_seq(const mpc_inst_t *in, mpc_work_t *w, const int *seq, float *E_out)
{
    float xa[MPC_MAXP * MPC_MAXZ], xb[MPC_MAXP * MPC_MAXZ];
    float *x = xa, *y = xb, *t, J = 0.0f, E = 0.0f;
    int b, k, n = in->np * in->nz;
    memcpy(x, in->c0, sizeof(float) * n);
    for (b = 0; b < in->Nu; ++b) {
        int u = seq[b];
        for (k = w->start[b]; k < w->start[b + 1]; ++k) {
            propagate(in, u, x, w->GD[u][k], y);
            J += stage_cost(in, y, k);
            t = x; x = y; y = t;
        }
        E += in->blen[b] * in->e_step[u];
        J += in->rho * in->blen[b] * in->e_step[u];
    }
    *E_out = E;
    return J + terminal(in, x);
}

static void dfs(const mpc_inst_t *in, mpc_work_t *w, int b, float J, float E)
{
    int u, k, n = in->np * in->nz;
    w->nodes++;
    if (b == in->Nu) {
        float Jt = J + terminal(in, w->xs[b]);
        w->leaves++;
        if (Jt < w->best_J) {
            w->best_J = Jt;
            memcpy(w->best_seq, w->prefix, sizeof(int) * in->Nu);
        }
        return;
    }
    for (u = 0; u < in->nu; ++u) {
        float E2 = E + in->blen[b] * in->e_step[u], J2;
        float xa[MPC_MAXP * MPC_MAXZ], *x = xa, *y = w->xs[b + 1], *t;
        if (E2 > in->e_allow) continue;
        J2 = J + in->rho * in->blen[b] * in->e_step[u];
        memcpy(x, w->xs[b], sizeof(float) * n);
        for (k = w->start[b]; k < w->start[b + 1]; ++k) {
            propagate(in, u, x, w->GD[u][k], y);
            J2 += stage_cost(in, y, k);
            t = x; x = y; y = t;
        }
        if (x != w->xs[b + 1]) memcpy(w->xs[b + 1], x, sizeof(float) * n);
        if (w->mode == MPC_BB_FORWARD) {
            if (J2 + bound_forward(in, w, w->xs[b + 1], w->start[b + 1]) >= w->best_J) continue;
        } else if (w->mode == MPC_BB_AFFINE) {
            if (J2 + bound_affine(in, w, w->xs[b + 1], w->start[b + 1]) >= w->best_J) continue;
        }
        w->prefix[b] = u;
        dfs(in, w, b + 1, J2, E2);
    }
}

int mpc_solve(const mpc_inst_t *in, int mode, mpc_work_t *w, int *seq_out, float *J_out)
{
    int u, b, n = in->np * in->nz, seq[MPC_MAXNU];
    if (in->nz > MPC_MAXZ || in->np > MPC_MAXP || in->nu > MPC_MAXU || in->N > MPC_MAXN || in->Nu > MPC_MAXNU)
        return -1;
    w->mode = mode;
    w->nodes = w->leaves = 0;
    prepare(in, w);
    w->best_J = FLT_MAX;
    /* incumbent from the constant sequences (the zero sequence always satisfies the budget) */
    if (mode != MPC_EXHAUSTIVE) {
        for (u = 0; u < in->nu; ++u) {
            float E, J;
            for (b = 0; b < in->Nu; ++b) seq[b] = u;
            J = eval_seq(in, w, seq, &E);
            if (E <= in->e_allow && J < w->best_J) {
                w->best_J = J;
                memcpy(w->best_seq, seq, sizeof(int) * in->Nu);
            }
        }
    }
    memcpy(w->xs[0], in->c0, sizeof(float) * n);
    dfs(in, w, 0, 0.0f, 0.0f);
    memcpy(seq_out, w->best_seq, sizeof(int) * in->Nu);
    *J_out = w->best_J;
    return 0;
}
