"""
Re-implemented fuzzy baselines on the same decoupled MIMO loop as the proposed
controller.

  method='sahin_ulu'  SI-IT2-FPID with online FOU tuning (OTM)
                      I. Sahin, C. Ulu, Int. J. Fuzzy Syst. 28(6):2263-2275, 2026
  method='kluska'     PID-like fuzzy controller (PID-FC) with an absolute-
                      stability sector and adaptation of the FC-2 consequents
                      J. Kluska, T. Zabinski, IEEE TFS 28(3):523-533, 2020
  method=None/'ours'  FuzzyPIDT2Env, unchanged

Shared with the proposed controller: plant, simplified decoupler, sampling
time, simulation length, actuator limits +-50, anti-windup, derivative filter
s/(tau_f s + 1), and the IMC-PID nominal design (the exact PID equivalent of
the type-1 controller).  Action a = 0 always gives the nominal IMC-PID, so the
GRPO reference policy means the same thing for every method.

Deviations from the papers (state these in the manuscript):
  * Both papers are SISO; each method is applied per loop after the decoupler.
  * Sahin & Ulu: eqs. (1)-(4), (14)-(16) are implemented.  Eq. (17) of the
    paper also replaces |e_hat| by |e_hat + de_hat| inside the mapping, which
    contradicts eq. (14); eq. (14) is followed here.  The derivative
    estimator is the shared filtered derivative.
  * Anti-windup: conditional integration for the position-form controllers
    (proposed with antiwindup='clamp', Sahin & Ulu); back-calculation with
    Tt = sqrt(Ti Td) for the velocity-form PID-FC (Kluska & Zabinski).
  * Kluska & Zabinski: FC-1 is discretised so the linear PID-FC equals the
    position-form PID with a filtered D term (see _FC1).  With FOPDT plants
    (relative degree 1) Corollary 1 is otherwise violated for a PID-FC; the
    filter keeps the loop strictly proper.  If the IMC-PID lies outside
    every certified sector, it is detuned into one (Step 1 redesign).
    The sector is computed per decoupled loop over common +-unc scaling of
    the plant; residual off-diagonal coupling (non-zero for the 4x4 plant and
    where decoupler delays are clamped) is ignored in that calculation.
    The reference-model consequents (Step 3) are tuned by GRPO instead of the
    exhaustive candidate search, which is infeasible for 16-32 consequents.
    No setpoint prefilter is used, so all methods see the same reference.
"""
import warnings
import numpy as np

from mimo_fuzzy_env import (MIMOPlant, SimplifiedDecoupler, TFDelay, pmul,
                            fit_fopdt)
from fuzzy_pid_t2 import FuzzyPIDT2Env, TAU_F


# ============================================================================
# factory
# ============================================================================
def make_env(method=None, **kw):
    """Drop-in replacement for FuzzyPIDT2Env(**kw)."""
    if method in (None, 'ours'):
        return FuzzyPIDT2Env(**kw)
    if method == 'sahin_ulu':
        return SahinUluEnv(**kw)
    if method == 'kluska':
        return KluskaEnv(**kw)
    raise ValueError(f"unknown method {method!r}")


# ============================================================================
# plant perturbations (shared by training, gamma selection and evaluation)
# ============================================================================
COMMON_PROB = 0.5


def _scale_tau(c, f):
    c = np.asarray(c, float)
    deg = len(c) - 1
    return np.array([c[i] * f ** (deg - i) for i in range(len(c))])


def perturb(spec, p):
    """Scale every gain, time constant and delay by (1 + p)."""
    f = 1.0 + p
    n = len(spec['g'])
    g = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            num, den, th = spec['g'][i][j]
            g[i][j] = (_scale_tau(num, f) * f, _scale_tau(den, f), th * f)
    return dict(spec, g=g)


def sampled_spec(spec, delta, plant_seed):
    """Random plant within +-delta: half scale everything together, half
    scale each parameter independently.  Identical to the DR training code."""
    rng = np.random.default_rng(plant_seed)
    n = len(spec['g'])
    if rng.random() < COMMON_PROB:
        f = np.full((n, n, 3), 1.0 + rng.uniform(-delta, delta))
    else:
        f = 1.0 + rng.uniform(-delta, delta, (n, n, 3))

    g = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            num, den, th = spec['g'][i][j]
            fk, ft, fl = f[i, j]
            g[i][j] = (_scale_tau(num, ft) * fk, _scale_tau(den, ft), th * fl)
    return dict(spec, g=g)


def _pw(a, nom, lo, hi):
    """Piecewise-linear map: a=-1 -> lo, a=0 -> nom, a=+1 -> hi."""
    a = np.clip(np.asarray(a, float), -1.0, 1.0)
    return np.where(a >= 0.0, nom + a * (hi - nom), nom + a * (nom - lo))


# ============================================================================
# common base: same plant / decoupler / IMC-PID as the proposed method
# ============================================================================
class _PIDBaseEnv(FuzzyPIDT2Env):
    def __init__(self, plant='wb', tau_f=TAU_F, tauC_factor=1.0,
                 fuzzy_type=None, tr_method=None, fix_fou=None, tune_fou=None,
                 **kw):
        # build the type-1 env only to reuse its IMC scaling factors
        super().__init__(plant=plant, fuzzy_type=1, tauC_factor=tauC_factor,
                         tau_f=tau_f, **kw)
        self.tau_f = tau_f
        self.pid0 = [c.equivalent_pid() for c in self.ctl]   # (Kp, Ki, Kd)
        self.ks = [1.0 / r if abs(r) > 1e-12 else 1.0 for r in self.R_STEP]

    def nominal_action(self):
        return np.zeros(self.n_act)


# ============================================================================
# Sahin & Ulu (2026): SI-IT2-FPID with online FOU tuning
# ============================================================================
SU_LO, SU_HI = 0.02, 0.98                   # bounds on k1, k2 and alpha, eq. (16)


def su_gamma(x, m0):
    """Eq. (15): nonlinear gain of the single-input IT2-FC, x = |e_hat|."""
    return 0.5 * (1.0 / (m0 + x - m0 * x) + (m0 - 1.0) / (m0 * x - 1.0))


class SahinUluPID:
    """e_hat = sat(ks e); alpha = sat(k1 |e_hat + de_hat| + k2);
    e_IT2 = e_hat * gamma(|e_hat|, alpha) / ks;  u = PID(e_IT2)."""

    def __init__(self, Kp, Ki, Kd, ks, k1, k2, ts, tau_f,
                 umin=-50.0, umax=50.0):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.ks, self.k1, self.k2 = ks, k1, k2
        self.ts, self.umin, self.umax = ts, umin, umax
        self.d_hat = TFDelay([1.0, 0.0], [tau_f, 1.0], 0.0, ts)   # derivative estimator
        self.d_it2 = TFDelay([1.0, 0.0], [tau_f, 1.0], 0.0, ts)   # D term
        self.reset()

    def reset(self):
        self.I = 0.0
        self.d_hat.reset()
        self.d_it2.reset()

    def step(self, e):
        eh = float(np.clip(self.ks * e, -1.0, 1.0))                  # eq. (1)
        deh = self.d_hat.step(eh)
        alpha = float(np.clip(self.k1 * abs(eh + deh) + self.k2,     # eq. (16)
                              SU_LO, SU_HI))
        e_it2 = eh * su_gamma(abs(eh), alpha) / self.ks              # eqs. (3), (14)
        de = self.d_it2.step(e_it2)

        # eq. (4), conditional-integration anti-windup: stop integrating only
        # when integration would push further into saturation.  (Back-
        # calculation lets a derivative kick wind the integrator the wrong way.)
        I_try = self.I + e_it2 * self.ts
        u = self.Kp * e_it2 + self.Ki * I_try + self.Kd * de
        push = self.Ki * e_it2
        if (u > self.umax and push > 0) or (u < self.umin and push < 0):
            I_try = self.I
            u = self.Kp * e_it2 + self.Ki * I_try + self.Kd * de
        self.I = I_try
        return float(min(max(u, self.umin), self.umax))


class SahinUluEnv(_PIDBaseEnv):
    """Action per loop: [k_p, k_i, k_d, k1, k2] with tune_pid=True,
    [k1, k2] with tune_pid=False (the paper's procedure: PID fixed).

    PID corrections: K_BOUND ** a, as for the proposed method.
    k1, k2: piecewise-linear in [0.02, 0.98] around K1_NOM, K2_NOM.
    K2_NOM = 0.618 gives gamma(0) = 1, i.e. the IMC-PID gain near setpoint.
    K1_NOM = 0.1 is the paper's Example 1 value.
    """
    K1_NOM = 0.1
    K2_NOM = (np.sqrt(5.0) - 1.0) / 2.0

    def __init__(self, tune_pid=True, **kw):
        super().__init__(**kw)
        self.tune_pid = tune_pid
        self.n_per_loop = 5 if tune_pid else 2
        self.n_act = self.n_loop * self.n_per_loop
        self.ctl = [SahinUluPID(*self.pid0[i], ks=self.ks[i],
                                k1=self.K1_NOM, k2=self.K2_NOM,
                                ts=self.ts, tau_f=self.tau_f)
                    for i in range(self.n_loop)]

    def decode(self, a):
        """action -> list of (Kp, Ki, Kd, k1, k2) per loop."""
        a = np.clip(np.asarray(a, float).ravel(), -1.0, 1.0)
        a = a.reshape(self.n_loop, self.n_per_loop)
        out = []
        for i in range(self.n_loop):
            Kp, Ki, Kd = self.pid0[i]
            if self.tune_pid:
                kp, ki, kd = self.K_BOUND ** a[i, :3]
                Kp, Ki, Kd = kp * Kp, ki * Ki, kd * Kd
            r = a[i, -2:]
            k1 = float(_pw(r[0], self.K1_NOM, SU_LO, SU_HI))
            k2 = float(_pw(r[1], self.K2_NOM, SU_LO, SU_HI))
            out.append((Kp, Ki, Kd, k1, k2))
        return out

    def _apply(self, a):
        if a is None:
            return
        for c, (Kp, Ki, Kd, k1, k2) in zip(self.ctl, self.decode(a)):
            c.Kp, c.Ki, c.Kd, c.k1, c.k2 = Kp, Ki, Kd, k1, k2

    def describe(self, a, prefix='  '):
        for i, (Kp, Ki, Kd, k1, k2) in enumerate(self.decode(a)):
            print(f"{prefix}loop {i+1}:  Kp={Kp:9.4f} Ki={Ki:9.5f} Kd={Kd:9.4f}"
                  f"   k1={k1:.3f} k2={k2:.3f}")


# ============================================================================
# Kluska & Zabinski (2020): PID-FC with absolute-stability sector
# ============================================================================
KZ_H = np.array([-12500.0, -0.75, -0.3, -0.05, 0.0,
                 0.05, 0.3, 0.75, 12500.0])         # partition of eq. (43)
KZ_GAMMA_GRID = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
KZ_KCAP = 10.0                                      # cap K at 10 kr if unbounded
_SECTOR_CACHE = {}


def _tf(num, den, theta, s):
    return (np.polyval(np.asarray(num, float), s)
            / np.polyval(np.asarray(den, float), s) * np.exp(-theta * s))


def effective_loops(spec_plant, spec_dec, s):
    """Diagonal of G(s) M(s): plant spec_plant, decoupler built from spec_dec
    exactly as SimplifiedDecoupler does (negative delays clamped to 0)."""
    gp, gd = spec_plant['g'], spec_dec['g']
    n = len(gp)
    G = [[_tf(*gp[i][j], s) for j in range(n)] for i in range(n)]
    M = [[None] * n for _ in range(n)]
    for i in range(n):
        nii, dii, tii = gd[i][i]
        for j in range(n):
            if i == j:
                M[i][j] = 1.0
                continue
            nij, dij, tij = gd[i][j]
            M[i][j] = _tf(-pmul(nij, dii), pmul(dij, nii),
                          max(tij - tii, 0.0), s)
    return [sum(G[i][j] * M[j][i] for j in range(n)) for i in range(n)]


def _winding(Lp, La, c):
    """Encirclements of c by the Nyquist plot.  Contour: -jW -> -j eps,
    right indentation around s = 0, +j eps -> +jW, closed at |s| = inf."""
    z = np.concatenate([np.conj(Lp[::-1]), La, Lp]) - c
    z = np.append(z, z[0])
    return int(np.round(np.angle(z[1:] / z[:-1]).sum() / (2.0 * np.pi)))


def kluska_sector(spec, ts, tau_f, pid, unc=0.15, n_unc=5, margin=0.10,
                  n_beta=80):
    """Steps 1-2: absolute stability sector [beta, beta+K] per loop.

    Linear part of the loop: L(s) = sgn * N(s) * g_eff(s) / s with
    N(s) = 1 + q1 s + q2 s^2/(tau_f s+1)   (FC-1, see _FC1).
    For each beta: 1 + beta L must be stable (Nyquist, GHC at beta) and
    K = -1 / min_w Re[L / (1 + beta L)]  (condition C4), for every plant
    in the uncertainty set.  L does not depend on kr, so:
      * if a certified sector contains kr (with margin), take the smallest
        such beta, as in the paper's example;
      * otherwise the linear PID is outside every certified sector, and
        Step 1 is redone: kr (hence Kp, Ki, Kd, q1 and q2 kept) is scaled
        down to the largest gain a certified sector contains (rho < 1).
    """
    key = (spec['name'], ts, tau_f, unc, n_unc, margin,
           tuple(np.round(np.ravel(pid), 12)))
    if key in _SECTOR_CACHE:
        return _SECTOR_CACHE[key]

    n = len(spec['g'])
    w = np.logspace(np.log10(1e-3 / spec['T_end']), np.log10(np.pi / ts), 40000)
    s_pos = 1j * w
    s_arc = w[0] * np.exp(1j * np.linspace(-np.pi / 2, np.pi / 2, 201))
    fs = np.linspace(-unc, unc, n_unc) if unc > 0 else np.array([0.0])
    geff = [(effective_loops(perturb(spec, f), spec, s_pos),
             effective_loops(perturb(spec, f), spec, s_arc)) for f in fs]
    g0 = effective_loops(spec, spec, np.array([1j * w[0]]))

    def nhat(s, q1, q2):
        return 1.0 + q1 * s + q2 * s ** 2 / (tau_f * s + 1.0)

    out = []
    for i in range(n):
        Kp, Ki, Kd = pid[i]
        sgn = 1.0 if Ki > 0 else -1.0
        kr, q1, q2 = abs(Ki), Kp / Ki, Kd / Ki
        if np.sign(g0[i][0].real) != sgn:
            warnings.warn(f"[{spec['name']}] loop {i+1}: sign of Ki differs "
                          f"from the effective loop gain")

        Ls = [(sgn * nhat(s_pos, q1, q2) * gp[i] / s_pos,
               sgn * nhat(s_arc, q1, q2) * ga[i] / s_arc) for gp, ga in geff]

        # certified (beta, K) pairs
        cands = []
        for b in kr * np.geomspace(0.005, 1.0 - margin, n_beta):
            ok, K = True, np.inf
            for Lp, La in Ls:
                if _winding(Lp, La, -1.0 / b) != 0:
                    ok = False
                    break
                m = (Lp / (1.0 + b * Lp)).real.min()
                K = min(K, np.inf if m >= 0 else -1.0 / m)
            if ok:
                cands.append((b, min(K, KZ_KCAP * kr)))

        # largest gain kr_new <= kr inside a certified sector with margin;
        # among sectors that keep kr itself, the smallest beta
        best = None
        for b, K in cands:
            kr_new = min(kr, (b + K) / (1.0 + margin))
            if b > kr_new * (1.0 - margin):
                continue
            if best is None or (kr_new, -b) > best[0]:
                best = ((kr_new, -b), b, K, kr_new)

        if best is None:
            warnings.warn(f"[{spec['name']}] loop {i+1}: no certified sector "
                          f"found; using an uncertified +-{margin:.0%} sector "
                          f"around kr")
            beta, K, kr_new, certified = (kr * (1.0 - margin),
                                          2.0 * margin * kr, kr, False)
        else:
            _, beta, K, kr_new = best
            certified = True
            if kr_new < kr:
                warnings.warn(f"[{spec['name']}] loop {i+1}: IMC-PID is outside "
                              f"every certified sector; PID detuned by "
                              f"x{kr_new / kr:.3f} (Step 1 redesign)")
        out.append(dict(sgn=sgn, kr=kr_new, q1=q1, q2=q2, beta=beta, K=K,
                        certified=certified, rho=kr_new / kr))

    _SECTOR_CACHE[key] = out
    return out


class _FC1:
    """h = x + q1 dx + q2 d2x, discretised so that the linear PID-FC is
    exactly the position-form PID of the other methods:

        INT kr h dt = Ki INT x dt + Kp x + Kd [s/(tau_f s+1)] x

    dx is the backward difference of x (no filter on the P path) and d2x the
    backward difference of the filtered derivative (same filter as the D
    term of the other controllers)."""

    def __init__(self, q1, q2, ts, tau_f):
        self.q1, self.q2, self.ts = q1, q2, ts
        self.df = TFDelay([1.0, 0.0], [tau_f, 1.0], 0.0, ts)
        self.reset()

    def reset(self):
        self.df.reset()
        self.x1 = 0.0
        self.d1 = 0.0

    def step(self, x):
        d = self.df.step(x)
        h = (x + self.q1 * (x - self.x1) / self.ts
             + self.q2 * (d - self.d1) / self.ts)
        self.x1, self.d1 = x, d
        return h


class KluskaPIDFC:
    """u = INT phi(h) dt,  h = e + q1 de + q2 d2e  (FC-1),
    phi = piecewise-linear through (H_k, Q_k)    (FC-2, eqs. (17)-(19)).
    Optional adaptation, eqs. (34)-(35) with sigma1 = sigma2 = 0."""

    def __init__(self, sec, H, ts, tau_f, umin=-50.0, umax=50.0,
                 dead_zone=1e-3):
        self.sgn, self.kr = sec['sgn'], sec['kr']
        self.q1, self.q2 = sec['q1'], sec['q2']
        self.beta, self.K = sec['beta'], sec['K']
        self.H = np.asarray(H, float)
        self.nz = np.flatnonzero(self.H != 0.0)
        lo = self.sgn * self.H * self.beta
        hi = self.sgn * self.H * (self.beta + self.K)
        self.QL, self.QH = np.minimum(lo, hi), np.maximum(lo, hi)
        self.lam0 = (self.kr - self.beta) / self.K      # lambda giving phi = kr h
        self.ts, self.umin, self.umax, self.dz = ts, umin, umax, dead_zone
        # back-calculation tracking time, Astrom & Hagglund: Tt = sqrt(Ti Td),
        # with Ti = Kp/Ki = q1 and Ti Td = Kd/Ki = q2
        self.Tt = max(np.sqrt(self.q2) if self.q2 > 0 else self.q1, ts)

        self.fc1 = _FC1(self.q1, self.q2, ts, tau_f)       # on e
        self.fc1_eps = _FC1(self.q1, self.q2, ts, tau_f)   # on eps (adaptation)
        self.adaptive, self.gamma, self.ym, self.r = False, 0.0, None, 0.0
        self.set_lam(np.full(len(self.nz), self.lam0))
        self.reset()

    def q_of(self, lam):
        Q = np.zeros_like(self.H)
        Q[self.nz] = self.sgn * self.H[self.nz] * (self.beta
                                                   + np.asarray(lam) * self.K)
        return Q

    def set_lam(self, lam):
        self.Q_init = self.q_of(lam)

    def reset(self):
        self.z, self.k = 0.0, 0
        self.Q = self.Q_init.copy()
        self.fc1.reset()
        self.fc1_eps.reset()

    def step(self, e):
        H = self.H
        h = self.fc1.step(e)
        hc = min(max(h, H[0]), H[-1])
        j = int(np.clip(np.searchsorted(H, hc, side='right') - 1, 0, len(H) - 2))
        m = (H[j + 1] - hc) / (H[j + 1] - H[j])          # M_j(h); M_{j+1} = 1 - m
        phi = m * self.Q[j] + (1.0 - m) * self.Q[j + 1]

        if self.adaptive:
            k = min(self.k, len(self.ym) - 1)
            eps = self.ym[k] - (self.r - e)               # eq. (33)
            he = self.fc1_eps.step(eps)
            if abs(h) >= self.dz:                         # eq. (34)
                g = self.ts * self.gamma * self.sgn * he
                self.Q[j] += g * m
                self.Q[j + 1] += g * (1.0 - m)
                self.Q[j:j + 2] = np.clip(self.Q[j:j + 2],       # eq. (35)
                                          self.QL[j:j + 2], self.QH[j:j + 2])
        # u = INT phi dt with back-calculation anti-windup.  A hard clamp on
        # the integral cuts the rise of a derivative kick but keeps its decay,
        # which slams u to the opposite limit; the tracking term avoids that.
        self.k += 1
        z = self.z + phi * self.ts
        u = min(max(z, self.umin), self.umax)
        self.z = z + (u - z) * min(self.ts / self.Tt, 1.0)
        return float(u)


class KluskaEnv(_PIDBaseEnv):
    """Action per loop: 8 values -> lambda_k in [0, 1] for the non-zero
    breakpoints, Q_k = sgn H_k (beta + lambda_k K).  a = 0 -> linear IMC-PID.

    adaptive=False : non-adaptive PID-FC (use for GRPO training = Step 3).
    adaptive=True  : reference model = tuned PID-FC on the NOMINAL plant,
                     adaptation of Q_k online (Steps 4-5).
                     q_init='pid' : Q starts at the linear PID (paper, Step 4)
                     q_init='ref' : Q starts at the reference model
    Adaptation weights w = gamma * (1, q1, q2), the ratios of the paper's
    example; gamma = gamma_factor * |kr| / (T + L) of the loop's FOPDT fit.
    """

    def __init__(self, adaptive=False, q_init='pid', gamma_factor=0.1,
                 unc=0.15, margin=0.10, dead_zone=1e-3, **kw):
        super().__init__(**kw)
        self.n_per_loop = len(KZ_H) - 1
        self.n_act = self.n_loop * self.n_per_loop
        self.adaptive, self.q_init = adaptive, q_init
        self.gamma_factor = gamma_factor

        self.sector = kluska_sector(self.spec, self.ts, self.tau_f, self.pid0,
                                    unc=unc, margin=margin)
        H = [KZ_H * abs(r) if abs(r) > 1e-12 else KZ_H for r in self.R_STEP]
        self.ctl = [KluskaPIDFC(self.sector[i], H[i], self.ts, self.tau_f,
                                dead_zone=dead_zone * max(abs(self.R_STEP[i]), 1e-12))
                    for i in range(self.n_loop)]
        self.ref_ctl = [KluskaPIDFC(self.sector[i], H[i], self.ts, self.tau_f)
                        for i in range(self.n_loop)]
        self.ref_plant = MIMOPlant(self.spec, self.ts)
        self.ref_dec = SimplifiedDecoupler(self.spec, self.ts, warn=False)
        self.T_loop = [sum(fit_fopdt(*self.spec['g'][i][i])[1:])
                       for i in range(self.n_loop)]

    def set_gamma_factor(self, c):
        self.gamma_factor = c

    def lambdas(self, a):
        a = np.clip(np.asarray(a, float).ravel(), -1.0, 1.0)
        a = a.reshape(self.n_loop, self.n_per_loop)
        return [_pw(a[i], c.lam0, 0.0, 1.0) for i, c in enumerate(self.ctl)]

    def _simulate_ref(self):
        """Reference model y_m on the nominal plant; ym[k] pairs with the
        y the main loop's controller sees at step k."""
        self.ref_plant.reset()
        self.ref_dec.reset()
        for c in self.ref_ctl:
            c.reset()
        n = self.n_loop
        n_step = int(round(self.T_END / self.ts))
        y, zero = np.zeros(n), np.zeros(n)
        ym = np.empty((n_step, n))
        for k in range(n_step):
            t = k * self.ts
            ym[k] = y
            r = self.R_STEP if t >= self.T_STEP else zero
            e = r - y
            v = np.array([self.ref_ctl[i].step(e[i]) for i in range(n)])
            u = self.ref_dec.step(v)
            if t >= self.D_TIME:
                u = u + self.D_MAG
            y = self.ref_plant.step(u)
            if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e4:
                ym[k + 1:] = ym[k]
                break
        return ym

    def _apply(self, a):
        if a is None:
            return
        lam = self.lambdas(a)
        if not self.adaptive:
            for c, l in zip(self.ctl, lam):
                c.set_lam(l)
                c.adaptive = False
            return

        for c, l in zip(self.ref_ctl, lam):
            c.set_lam(l)
        ym = self._simulate_ref()
        for i, c in enumerate(self.ctl):
            c.set_lam(lam[i] if self.q_init == 'ref'
                      else np.full(len(c.nz), c.lam0))
            c.adaptive = True
            c.ym = ym[:, i]
            c.r = self.R_STEP[i]
            c.gamma = self.gamma_factor * c.kr / self.T_loop[i]

    def describe(self, a, prefix='  '):
        for i, (c, l) in enumerate(zip(self.ctl, self.lambdas(a))):
            sec = self.sector[i]
            cert = "certified" if sec['certified'] else "NOT certified"
            if sec['certified'] and sec['rho'] < 1.0:
                cert += f", PID detuned x{sec['rho']:.3f}"
            print(f"{prefix}loop {i+1}:  kr={c.sgn * c.kr:9.5f} q1={c.q1:8.3f} "
                  f"q2={c.q2:9.3f}  sector [{c.beta:.4g}, {c.beta + c.K:.4g}] "
                  f"({cert})")
            print(f"{prefix}         lambda = {np.array2string(l, precision=3)}"
                  f"   (linear PID: {c.lam0:.3f})")


def kluska_select_gamma(env_kw, a, q_init='pid', delta=0.15,
                        factors=KZ_GAMMA_GRID,
                        val_seeds=tuple(range(1000, 1008))):
    """Step 5: pick the adaptation gain on the validation plants (nominal +
    the same 8 random plants used for DR validation).  gamma = 0 (no
    adaptation) is in the grid, so the choice can switch adaptation off."""
    env = make_env(**{**env_kw, 'method': 'kluska', 'adaptive': True,
                      'q_init': q_init})
    nominal = env.plant
    plants = [nominal] + [MIMOPlant(sampled_spec(env.spec, delta, s), env.ts)
                          for s in val_seeds]
    scores = {}
    for c in factors:
        env.set_gamma_factor(c)
        r = []
        for p in plants:
            env.plant = p
            r.append(env.rollout(a))
        scores[c] = float(np.mean(r))
    env.plant = nominal
    best = max(scores, key=scores.get)
    return best, scores


# ============================================================================
# self-test:  python baselines.py
# ============================================================================
if __name__ == "__main__":
    from mimo_fuzzy_env import PLANTS

    x = np.linspace(0.0, 1.0, 11)
    for m0 in (0.05, 0.5, 0.95):
        g = su_gamma(x, m0)
        print(f"Sahin-Ulu gamma, m0={m0}: gamma(0)={g[0]:.4f} "
              f"(closed form {0.5 * (1 / m0 + 1 - m0):.4f}), gamma(1)={g[-1]:.4f}")

    TAU_F_OF = {'hvac': 2.0, 'chen': 0.1}
    for key in ('vl', 'wb', 'ww', 'isp', 'hvac'):
        tau_f = max(TAU_F_OF.get(key, 0.01), 4.0 * PLANTS[key]['ts'])
        kw = dict(plant=key, R_step=1.0, warn=False, tau_f=tau_f)
        ours = make_env(fuzzy_type=1, **kw)
        ours_c = make_env(fuzzy_type=1, antiwindup='clamp', **kw)
        fix_c = make_env(fuzzy_type=2, antiwindup='clamp', **kw)
        fix_b = make_env(fuzzy_type=2, **kw)
        su = make_env(method='sahin_ulu', **kw)
        kz = make_env(method='kluska', **kw)
        print(f"\n{PLANTS[key]['name']}")
        print(f"  nominal IAE, back-calc : type-1 "
              f"{-ours.rollout(ours.nominal_action()):9.3f} | fixed FOU "
              f"{-fix_b.rollout(fix_b.nominal_action()):9.3f}")
        print(f"  nominal IAE, clamp     : type-1 "
              f"{-ours_c.rollout(ours_c.nominal_action()):9.3f} | fixed FOU "
              f"{-fix_c.rollout(fix_c.nominal_action()):9.3f}")
        print(f"  nominal IAE, baselines : Sahin-Ulu "
              f"{-su.rollout(su.nominal_action()):9.3f} | Kluska "
              f"{-kz.rollout(kz.nominal_action()):9.3f}")
        kz.describe(kz.nominal_action())

        a = np.random.default_rng(0).uniform(-0.5, 0.5, kz.n_act)
        kz_ref = make_env(method='kluska', adaptive=True, q_init='ref', **kw)
        d = abs(kz.rollout(a) - kz_ref.rollout(a))
        print(f"  adaptive (init ref) == non-adaptive on nominal plant: "
              f"diff {d:.2e}")
