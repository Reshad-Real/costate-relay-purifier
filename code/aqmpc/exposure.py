"""Exposure index and closed-loop performance metrics."""
import numpy as np
from .params import PHI, PSI, G_NORM, C_BOUND2, C_BOUND3, iPM25, iPM10


def weight_tensor(phi=PHI, psi=PSI, g=G_NORM):
    return (np.asarray(phi) / np.asarray(g))[:, None] * np.asarray(psi)[None, :]   # (P,Z)


def index_series(C, phi=PHI, psi=PSI, g=G_NORM):
    """Instantaneous health-weighted index H(t) for C of shape (n,P,Z)."""
    return np.einsum("npz,pz->n", np.maximum(C, 0.0), weight_tensor(phi, psi, g))


def metrics(C, U, occ, dt, fan, phi=PHI, psi=PSI, g=G_NORM, alt_weights=None):
    """C (n,P,Z) true concentrations, U (n,) applied duty, occ (n,) occupied mask, dt in h.
    alt_weights: optional list of (phi, psi) pairs; CE under each is returned as CE_alt."""
    H = index_series(C, phi, psi, g)
    w = occ * dt
    hours = w.sum()
    pm25_w = np.einsum("nz,z->n", C[:, iPM25, :], np.asarray(psi))
    exceed = C > C_BOUND3[None, :, None]
    exceed2 = C > C_BOUND2[None, :, None]
    any_pm = exceed2[:, [iPM25, iPM10], :].any(axis=(1, 2))
    any_all = exceed.any(axis=(1, 2))
    starts = np.diff(np.concatenate([[0], any_all.astype(int)])) == 1
    P = fan.power(U)
    sw = np.count_nonzero(np.diff(U) != 0)
    return {
        "CE": float((H * w).sum()),
        "TWA_PM25": float((pm25_w * w).sum() / hours),
        "TWA_PM10": float((np.einsum("nz,z->n", C[:, iPM10, :], np.asarray(psi)) * w).sum() / hours),
        "peak_PM25": float(C[:, iPM25, :].max()),
        "exceed_L2_min_PM": float((any_pm * w).sum() * 60),
        "exceed_L3_min": float((any_all * w).sum() * 60),
        "exceed_L3_episodes": int((starts & (occ > 0)).sum()),
        "exceed_L3_area": float(((np.maximum(C - C_BOUND3[None, :, None], 0)
                                  / C_BOUND3[None, :, None]).sum(axis=(1, 2)) * w).sum()),
        "energy_Wh": float((P * dt).sum()),
        "runtime_h": float(((U > 0) * dt).sum()),
        "switches": int(sw),
        **({"CE_alt": [float((index_series(C, ph, ps, g) * w).sum()) for ph, ps in alt_weights]}
           if alt_weights else {}),
    }
