#include <stdio.h>
#include "costate_params.h"
int main(void) {
    FILE *f = fopen("costate_cases.txt", "r"); int n = 0, bad = 0;
    float occ, rho, e, x[CS_NP][CS_NZ];
    while (fscanf(f, "%f %f", &occ, &rho) == 2) {
        for (int p = 0; p < CS_NP; ++p) for (int j = 0; j < CS_NZ; ++j) fscanf(f, "%f", &x[p][j]);
        fscanf(f, "%f", &e);
        int u = costate_duty(&COSTATE_PARAMS, (const float (*)[CS_NZ])x, occ, rho);
        if (u != (int)e) bad++;
        n++;
    }
    printf("%d cases, %d mismatches\n", n, bad);
    return 0;
}
