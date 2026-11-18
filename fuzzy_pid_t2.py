"""
Type-1 / type-2 fuzzy PID, following the MathWorks "Fuzzy PID Control with
Type-2 FIS" example (adapted from Mendel, Uncertain Rule-Based Fuzzy Systems,
2nd ed., pp. 229-234, 600-608).

Structure -- the FIS output IS the control action:

    E  = Ce * e                        clipped to [-1,1]
    dE = Cd * F(s)e,  F(s)=100s/(s+100) = s/(tau_f s + 1),  tau_f = 0.01
    U  = FIS(E, dE)                    in [-1,1]
    u  = C0 * INT(U) dt  +  C1 * U

FIS: 3 triangular MFs per input (N,Z,P at -1,0,1, half-width 1), 9 rules,
zero-order Sugeno, singleton consequents {-1,-0.5,0,0.5,1}:

    dE:      N     Z     P
  E=N    -1.0  -0.5   0.0
  E=Z    -0.5   0.0   0.5
  E=P     0.0   0.5   1.0

c(i,j) = (E_i + dE_j)/2, so with partition-of-unity MFs the type-1 FIS is
EXACTLY U = (E + dE)/2 -- a linear PID with

    Kp = (C0*Cd + C1*Ce)/2,   Ki = C0*Ce/2,   Kd = C1*Cd/2

The FOU is what makes the type-2 control surface nonlinear.

Type-2: upper MF is the nominal triangle; lower MF is height-scaled by
LowerScale s and support-narrowed by LowerLag lam:

    upper_k(x) = tri(x; c_k, halfwidth 1)
    lower_k(x) = s_k * tri(x; c_k, halfwidth 1 - lam_k)

MATLAB example values:  scale = [0.2 0.9 0.2; 0.3 0.9 0.3], lag = 0
                        (alternative: scale = 0.7, lag = 0.1)

Type reduction: EIASC (exact Karnik-Mendel, as MATLAB) or Nie-Tan.

Scaling factors from a FOPDT fit (K, T, L) of the diagonal element:
    Ce = 1/r                      Cd = min(T, L/2) * Ce
    C0 = 1/(K*Ce*(tauC + L/2))    C1 = max(T, L/2) * C0
"""
import warnings
import numpy as np

from mimo_fuzzy_env import MIMOFuzzyEnv, PLANTS, fit_fopdt, TFDelay

TAU_F = 0.01                       # 100s/(s+100)

# ---------------------------------------------------------------- FIS constants
MFC = np.array([-1.0, 0.0, 1.0])                  # MF centres: N, Z, P
UTAB = 0.5 * (MFC[:, None] + MFC[None, :])        # c(i,j) = (E_i + dE_j)/2
_ORD = np.argsort(UTAB.ravel())
_USORT = UTAB.ravel()[_ORD]

# MathWorks example FOU
MATLAB_SCALE = np.array([[0.2, 0.9, 0.2],         # input E
                         [0.3, 0.9, 0.3]])        # input dE
MATLAB_LAG = np.zeros((2, 3))


def tri(x, halfwidth):
    """Triangular MFs centred on MFC."""
    return np.clip(1.0 - np.abs(x - MFC) / halfwidth, 0.0, 1.0)


def eiasc(f_lo, f_hi):
    """Enhanced Iterative Algorithm with Stop Condition -> (y_left, y_right)."""
    lo, hi, y = f_lo.ravel()[_ORD], f_hi.ravel()[_ORD], _USORT
    n = len(y)
    d = hi - lo
    base_a, base_b = float(lo @ y), float(lo.sum())

    a, b, L, yl = base_a, base_b, 0, y[0]
    while L < n:
        a += y[L] * d[L]
        b += d[L]
        yl = a / b if b > 1e-12 else y[L]
        L += 1
        if L >= n or yl <= y[L]:
            break

    a, b, R, yr = base_a, base_b, n - 1, y[-1]
    while R >= 0:
        a += y[R] * d[R]
        b += d[R]
        yr = a / b if b > 1e-12 else y[R]
        R -= 1
        if R < 0 or yr >= y[R]:
            break

    return yl, yr


class FuzzyPIDT2:
    """Type-1 / interval type-2 fuzzy PID, FIS-as-controller."""

    def __init__(self, Ce, Cd, C0, C1, ts, umin=-50.0, umax=50.0,
                 fuzzy_type=1, tr_method='eiasc', tau_f=TAU_F, warn=True,
                 antiwindup='backcalc'):
        self.antiwindup = antiwindup     # 'backcalc' (original) | 'clamp'
        self.Ce0, self.Cd0, self.C00, self.C10 = Ce, Cd, C0, C1
        self.Ce, self.Cd, self.C0, self.C1 = Ce, Cd, C0, C1
        self.ts, self.umin, self.umax = ts, umin, umax
        self.fuzzy_type = fuzzy_type
        self.tr_method = tr_method

        if warn and tau_f < 2.0 * ts:
            warnings.warn(
                f"derivative filter tau_f={tau_f} < 2*ts={2*ts}: the filter pole "
                f"is faster than the sample rate, Tustin will ring. Use a finer "
                f"ts or a larger tau_f.")
        self.dfilt = TFDelay([1.0, 0.0], [tau_f, 1.0], 0.0, ts)   # s/(tau_f s+1)

        # FOU: LowerScale / LowerLag, row 0 = input E, row 1 = input dE
        self.scale = np.ones((2, 3))
        self.lag = np.zeros((2, 3))
        self.reset()

    def reset(self):
        self.I = 0.0
        self.dfilt.reset()

    def set_matlab_fou(self):
        self.scale = MATLAB_SCALE.copy()
        self.lag = MATLAB_LAG.copy()
        return self

    def equivalent_pid(self):
        """Exact PID gains of the type-1 controller."""
        return (0.5 * (self.C0 * self.Cd + self.C1 * self.Ce),
                0.5 * self.C0 * self.Ce,
                0.5 * self.C1 * self.Cd)

    def fis(self, E, dE):
        up_e, up_d = tri(E, 1.0), tri(dE, 1.0)
        if self.fuzzy_type == 1:
            return float((np.outer(up_e, up_d) * UTAB).sum())   # weights sum to 1

        lo_e = np.minimum(self.scale[0] * tri(E, 1.0 - self.lag[0]), up_e)
        lo_d = np.minimum(self.scale[1] * tri(dE, 1.0 - self.lag[1]), up_d)
        f_hi, f_lo = np.outer(up_e, up_d), np.outer(lo_e, lo_d)

        if self.tr_method == 'nietan':
            Wn = f_lo + f_hi
            den = Wn.sum()
            return float((Wn * UTAB).sum() / den) if den > 1e-12 else 0.0
        yl, yr = eiasc(f_lo, f_hi)
        return 0.5 * (yl + yr)

    def step(self, e):
        de = self.dfilt.step(e)                      # filtered derivative
        E = float(np.clip(self.Ce * e, -1.0, 1.0))
        dE = float(np.clip(self.Cd * de, -1.0, 1.0))
        U = self.fis(E, dE)

        I_try = self.I + U * self.ts
        u = self.C0 * I_try + self.C1 * U
        if self.antiwindup == 'clamp':
            # conditional integration: stop integrating only when that would
            # push further into saturation
            push = self.C0 * U
            if (u > self.umax and push > 0) or (u < self.umin and push < 0):
                I_try = self.I
                u = self.C0 * I_try + self.C1 * U
            u = min(max(u, self.umin), self.umax)
        elif u > self.umax or u < self.umin:         # back-calculation (original)
            u = min(max(u, self.umin), self.umax)
            if abs(self.C0) > 1e-12:
                I_try = (u - self.C1 * U) / self.C0
        self.I = I_try
        return float(u)


# ---------------------------------------------------------------- scaling design
def scaling_factors(spec, loop, r_step=1.0, tauC_factor=1.0):
    """IMC nominal scaling factors, Eqns. (9)-(11) of the paper.

        Ce = 1                                   (fixed to unity)
        Cd = min(T, L/2) * Ce
        C0 = 1 / (K * Ce * (tauC + L/2)),  tauC = tauC_factor * L  (paper: ~L)
        C1 = max(T, L/2) * C0

    The sign of C0/C1 follows the sign of K automatically, so no manual sign
    assignment is needed for loops with a negative diagonal gain.
    """
    K, T, L = fit_fopdt(*spec['g'][loop][loop])
    Ce = 1.0                                             # paper: Ce fixed to 1
    Cd = min(T, L / 2.0) * Ce
    C0 = 2.0 / (K * Ce * (tauC_factor * L + L / 2.0))
    C1 = max(T, L / 2.0) * C0
    return Ce, Cd, C0, C1


# ---------------------------------------------------------------- environment
class FuzzyPIDT2Env(MIMOFuzzyEnv):
    """Paper-faithful IT2 fuzzy PID tuning environment.

    The FIS (interval type-2 membership functions + rule base) is FIXED at the
    MathWorks / Mendel FOU throughout tuning.  All plant-specific adaptation is
    realised through the output scaling factors, exactly as in the paper.

    Action (3 per loop, n_act = 6):

        a[3i+0] -> k_d^i      correction on Cd
        a[3i+1] -> k_0^i      correction on C0
        a[3i+2] -> k_1^i      correction on C1

    Each a in [-1, 1] maps logarithmically to k = K_BOUND ** a, so
    a = -1 -> 1/20 = 0.05,   a = 0 -> 1 (nominal design),   a = +1 -> 20.
    Ce is fixed to its nominal value (unity) and is NOT tuned.
    """

    K_BOUND = 20.0          # correction-factor bounds are [1/K_BOUND, K_BOUND]

    # FOU search (tune_fou=True), N and P share one value so U stays odd.
    # Order per loop: [outer E, Z E, outer dE, Z dE]
    S_NOM = np.array([0.2, 0.9, 0.3, 0.9])    # a = 0  -> MathWorks FOU
    S_LOW = np.array([0.05, 0.3, 0.05, 0.3])  # a = -1
    S_HIGH = 1.0                              # a = +1 -> type-1 surface

    def __init__(self, plant='wb', fuzzy_type=2, tauC_factor=1.0,
                 tr_method='eiasc', tau_f=TAU_F, fix_fou=True, tune_fou=False,
                 antiwindup='backcalc', **kw):
        if tune_fou and fuzzy_type != 2:
            raise ValueError("tune_fou=True needs fuzzy_type=2")
        kw.pop('fuzzy_type', None)
        warn = kw.get('warn', True)
        super().__init__(plant=plant, **kw)

        self.fuzzy_type = fuzzy_type
        self.tune_fou = tune_fou
        self.n_per_loop = 7 if tune_fou else 3
        self.n_act = self.n_loop * self.n_per_loop

        self.scaling = [scaling_factors(self.spec, i, self.R_STEP[i], tauC_factor)
                        for i in range(self.n_loop)]
        self.ctl = [FuzzyPIDT2(*self.scaling[i], ts=self.ts,
                               fuzzy_type=fuzzy_type, tr_method=tr_method,
                               tau_f=tau_f, warn=warn and i == 0,
                               antiwindup=antiwindup)
                    for i in range(self.n_loop)]

        # FIS held fixed for the whole tuning run
        if fuzzy_type == 2 and fix_fou:
            for c in self.ctl:
                c.set_matlab_fou()

        self.gains = [c.equivalent_pid()[:2] for c in self.ctl]

    # ------------------------------------------------------------- decoding
    def corrections(self, a):
        """action -> array (2, 3) of correction factors [k_d, k_0, k_1]."""
        a = np.clip(np.asarray(a, float).ravel(), -1.0, 1.0)
        return self.K_BOUND ** a.reshape(self.n_loop, self.n_per_loop)[:, :3]

    def decode_fou(self, a):
        """action -> array (n_loop, 4) of lower scales; a=0 gives S_NOM."""
        a = np.clip(np.asarray(a, float).ravel(), -1.0, 1.0)
        a = a.reshape(self.n_loop, self.n_per_loop)[:, 3:]
        return np.where(a >= 0.0,
                        self.S_NOM + a * (self.S_HIGH - self.S_NOM),
                        self.S_NOM + a * (self.S_NOM - self.S_LOW))

    def decode(self, a):
        """action -> list of (Ce, Cd, C0, C1) per loop.  Single source of truth."""
        k = self.corrections(a)
        out = []
        for i in range(self.n_loop):
            Ce, Cd0, C00, C10 = self.scaling[i]
            out.append((Ce, k[i, 0] * Cd0, k[i, 1] * C00, k[i, 2] * C10))
        return out

    def _apply(self, a):
        if a is None:
            return
        for c, (Ce, Cd, C0, C1) in zip(self.ctl, self.decode(a)):
            c.Ce, c.Cd, c.C0, c.C1 = Ce, Cd, C0, C1
        if self.tune_fou:
            for c, (se_o, se_z, sd_o, sd_z) in zip(self.ctl, self.decode_fou(a)):
                c.scale = np.array([[se_o, se_z, se_o], [sd_o, sd_z, sd_o]])

    # ------------------------------------------------------------- helpers
    def nominal_action(self):
        """All correction factors = 1, i.e. the pure IMC nominal design."""
        return np.zeros(self.n_act)

    # backwards-compatible aliases (both now mean "nominal gains")
    type1_action = nominal_action
    matlab_fou_action = nominal_action

    def describe(self, a, prefix='  '):
        for i, (Ce, Cd, C0, C1) in enumerate(self.decode(a)):
            k = self.corrections(a)[i]
            Kp = 0.5 * (C0 * Cd + C1 * Ce)
            Ki = 0.5 * C0 * Ce
            Kd = 0.5 * C1 * Cd
            print(f"{prefix}loop {i+1}:  kd={k[0]:7.3f} k0={k[1]:7.3f} k1={k[2]:7.3f}")
            print(f"{prefix}         Ce={Ce:9.4f} Cd={Cd:9.4f} "
                  f"C0={C0:9.4f} C1={C1:9.4f}")
            print(f"{prefix}         equiv PID (type-1 surface): "
                  f"Kp={Kp:9.4f} Ki={Ki:9.5f} Kd={Kd:9.4f}")
            if self.tune_fou:
                se_o, se_z, sd_o, sd_z = self.decode_fou(a)[i]
                print(f"{prefix}         s_l: E(N,P)={se_o:.3f} E(Z)={se_z:.3f} "
                      f"dE(N,P)={sd_o:.3f} dE(Z)={sd_z:.3f}")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1, 1, (2000, 2))

    c1 = FuzzyPIDT2(1, 1, 1, 1, ts=0.1, fuzzy_type=1, warn=False)
    e1 = max(abs(c1.fis(*p) - 0.5 * (p[0] + p[1])) for p in pts)
    print(f"type-1 FIS == (E+dE)/2          max err {e1:.2e}")

    c2 = FuzzyPIDT2(1, 1, 1, 1, ts=0.1, fuzzy_type=2, warn=False)
    e2 = max(abs(c2.fis(*p) - 0.5 * (p[0] + p[1])) for p in pts)
    print(f"type-2 (scale=1,lag=0) collapse max err {e2:.2e}")

    c2.set_matlab_fou()
    km = np.array([c2.fis(*p) for p in pts[:500]])
    c2.tr_method = 'nietan'
    nt = np.array([c2.fis(*p) for p in pts[:500]])
    lin = 0.5 * (pts[:500, 0] + pts[:500, 1])
    print(f"MATLAB FOU: nonlinearity vs linear surface  {np.abs(km-lin).max():.4f}")
    print(f"            EIASC vs Nie-Tan                {np.abs(km-nt).max():.4f}")

    print()
    for key in ('wb', 'ww', 'vl', 'isp'):
        env = FuzzyPIDT2Env(plant=key, fuzzy_type=2, warn=False)
        m = env.metrics(env.nominal_action())
        Ce, Cd, C0, C1 = env.decode(env.nominal_action())[0]
        print(f"{PLANTS[key]['name'][:30]:30s} ts={env.ts:5.3f} n_act={env.n_act} "
              f"| nominal IAE {m['iae_tot']:9.3f} ISE {m['ise_tot']:9.3f} "
              f"| loop1 Cd={Cd:.4f} C0={C0:.4f} C1={C1:.4f}")