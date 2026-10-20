/* Exact solver for the blocked discrete exposure-economic MPC problem (portable C99, float).
 *
 * Decision: v in U^Nu (one duty per move block). Dynamics per pollutant p (block diagonal):
 *     x_{k+1,p} = A[u_k][p] x_{k,p} + G[u_k][p] d_{k,p}
 * Cost: sum_{k=1..N} dt*( occ_k <W,x_k> + M <1,(x_k-cb)_+/cb> ) + rho*sum_k e_step[u_k] + <Wf,x_N>
 * Hard constraint: sum_k e_step[u_k] <= e_allow.
 *
 * Modes: MPC_EXHAUSTIVE  - depth-first enumeration, energy pruning only
 *        MPC_BB_FORWARD  - branch and bound, bound = cost of the positive comparison system
 *                          x+ = A_lb x + G_lb d (A_lb = min_u A(u), elementwise), incl. penalty
 *        MPC_BB_AFFINE   - branch and bound, bound = affine cost-to-go of the comparison system
 *                          without the (non-negative) penalty term; lambda_k, beta_k by backward
 *                          recursion once per solve, O(Z) per bound evaluation per pollutant.
 * Both bounds are admissible because A(u), G(u) >= 0 elementwise (Metzler generator), x, d >= 0
 * and the stage/terminal costs are non-decreasing in x.
 */
#ifndef MPC_SOLVER_H
#define MPC_SOLVER_H

#if defined(ESP32) || defined(MPC_DEPLOYED_DIMS)
/* deployed dimensions: 3 zones, 6 pollutants, 4 duties, N = 30, Nu = 6 (workspace ~12 kB) */
#define MPC_MAXZ 3
#define MPC_MAXP 6
#define MPC_MAXU 4
#define MPC_MAXN 30
#define MPC_MAXNU 6
#else
#define MPC_MAXZ 8
#define MPC_MAXP 6
#define MPC_MAXU 6
#define MPC_MAXN 64
#define MPC_MAXNU 10
#endif

enum { MPC_EXHAUSTIVE = 0, MPC_BB_FORWARD = 1, MPC_BB_AFFINE = 2 };

typedef struct {
    int nz, np, nu, N, Nu;
    int blen[MPC_MAXNU];
    float dt, rho, M, e_allow;
    float e_step[MPC_MAXU];
    const float *A;    /* [nu][np][nz][nz] */
    const float *G;    /* [nu][np][nz][nz] */
    const float *W;    /* [np][nz]  index weights (phi/g * psi) */
    const float *Wf;   /* [np][nz]  terminal weights (may be NULL) */
    const float *cb;   /* [np][nz]  soft bounds */
    const float *D;    /* [N][np][nz] drive forecast */
    const float *occ;  /* [N] occupancy of predicted steps 1..N */
    const float *c0;   /* [np][nz] initial state */
} mpc_inst_t;

typedef struct {
    float GD[MPC_MAXU][MPC_MAXN][MPC_MAXP * MPC_MAXZ];
    float GlbD[MPC_MAXN][MPC_MAXP * MPC_MAXZ];
    float Alb[MPC_MAXP][MPC_MAXZ][MPC_MAXZ];
    float lam[MPC_MAXN + 1][MPC_MAXP * MPC_MAXZ];
    float beta[MPC_MAXN + 1];
    float xs[MPC_MAXNU + 1][MPC_MAXP * MPC_MAXZ];
    int prefix[MPC_MAXNU];
    int start[MPC_MAXNU + 1];
    int best_seq[MPC_MAXNU];
    float best_J;
    long nodes, leaves;
    int mode;
} mpc_work_t;

/* Solve; returns 0 on success. seq_out[Nu] receives the optimal blocked sequence (duty indices). */
int mpc_solve(const mpc_inst_t *in, int mode, mpc_work_t *w, int *seq_out, float *J_out);

#endif
