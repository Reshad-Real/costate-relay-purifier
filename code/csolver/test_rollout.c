#include <stdio.h>
#include <math.h>
#include "rollout_params.h"
int main(void) {
    FILE *f = fopen("rollout_cases.txt", "r"); int n = 0, bad = 0;
    float kf, rem, e, x[RL_NP][RL_NZ], d[RL_NP][RL_NZ];
    while (fscanf(f, "%f %f", &kf, &rem) == 2) {
        for (int p = 0; p < RL_NP; ++p) for (int j = 0; j < RL_NZ; ++j) fscanf(f, "%f", &x[p][j]);
        for (int p = 0; p < RL_NP; ++p) for (int j = 0; j < RL_NZ; ++j) fscanf(f, "%f", &d[p][j]);
        fscanf(f, "%f", &e);
        int u = rollout_duty(&ROLLOUT_PARAMS, (const float (*)[RL_NZ])x, (const float (*)[RL_NZ])d, (int)kf, rem);
        if (u != (int)e) { bad++; printf("mismatch case %d: C %d, Python %d\n", n, u, (int)e); }
        n++;
    }
    printf("%d cases, %d mismatches\n", n, bad);
    return 0;
}
