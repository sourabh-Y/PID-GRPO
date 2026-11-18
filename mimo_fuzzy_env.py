"""
General 2x2 MIMO fuzzy-PI RL environment.

Plant engine is a general rational transfer function with delay, so every
element is simulated exactly as written -- no FOPDT reduction anywhere in
the simulation path.

Plants:
    'wb'   Wood-Berry (WB) column
    'ww'   Wardle and Wood (WW) column
    'vl'   Vinante and Luyben (VL) column
    'isp'  Industrial Scale Polymerization (ISP) process

Decoupler is ideal/inverted:  D = [1, -g12/g11 ; -g21/g22, 1], built
symbolically from the polynomial data.  If a branch requires a NEGATIVE
delay (non-causal), the delay term is dropped and the rational part is
kept -- this happens on VL (d12) and ISP (d21).

Only numpy required.
"""
import numpy as np
import warnings

TS = 0.1

# ============================================================================
# polynomial helpers  (coefficients in DESCENDING powers of s)
# ============================================================================
def pmul(*ps):
    out = np.array([1.0])
    for p in ps:
        out = np.convolve(out, np.asarray(p, float))
    return out


def foh(tau):
    """first-order factor (tau s + 1)"""
    return np.array([tau, 1.0])


# ============================================================================
# discrete blocks
# ============================================================================
class Delay:
    """Transport delay with linear interpolation for the fractional part."""

    def __init__(self, theta, ts=TS):
        d = max(theta, 0.0) / ts
        self.n = int(np.floor(d))
        self.f = d - self.n
        self.L = self.n + 2
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.L)
        self.i = 0

    def step(self, u):
        self.buf[self.i] = u
        i0 = (self.i - self.n) % self.L
        i1 = (self.i - self.n - 1) % self.L
        y = (1.0 - self.f) * self.buf[i0] + self.f * self.buf[i1]
        self.i = (self.i + 1) % self.L
        return y


def bilinear(num, den, ts):
    """Tustin discretization. num/den descending in s -> bz/az ascending in z^-1."""
    num = np.atleast_1d(np.asarray(num, float))
    den = np.atleast_1d(np.asarray(den, float))
    N = max(len(num), len(den)) - 1
    K = 2.0 / ts

    def convert(c):
        out = np.zeros(N + 1)
        deg = len(c) - 1
        for idx, coef in enumerate(c):
            p = deg - idx                       # power of s
            if coef == 0.0:
                continue
            a = np.array([1.0])
            for _ in range(p):                  # (1 - z^-1)^p
                a = np.convolve(a, [1.0, -1.0])
            for _ in range(N - p):              # (1 + z^-1)^(N-p)
                a = np.convolve(a, [1.0, 1.0])
            out += coef * (K ** p) * a
        return out

    bz, az = convert(num), convert(den)
    return bz / az[0], az / az[0]


class TFDelay:
    """num(s)/den(s) * e^{-theta s}, Tustin-discretized IIR + delay buffer."""

    def __init__(self, num, den, theta, ts=TS):
        self.b, self.a = bilinear(num, den, ts)
        self.delay = Delay(theta, ts)
        self.reset()

    def reset(self):
        self.x = np.zeros(len(self.b))
        self.y = np.zeros(len(self.a))
        self.delay.reset()

    def step(self, u):
        ud = self.delay.step(u)
        self.x[1:] = self.x[:-1]
        self.x[0] = ud
        out = float(self.b @ self.x - self.a[1:] @ self.y[:-1])
        self.y[1:] = self.y[:-1]
        self.y[0] = out
        return out


# ============================================================================
# plant definitions   g[i][j] = (num, den, theta) here you can also measure or edit T final for simulation
# ============================================================================
PLANTS = {}

# --- A. Wood-Berry column ---------------------------------------------------
PLANTS['wb'] = dict(
    name='Wood-Berry (WB) column',
    T_end=300.0, ts=0.1,
    g=[[([12.8], foh(16.7), 1.0),
        ([-18.9], foh(21.0), 3.0)],
       [([6.6], foh(10.9), 7.0),
        ([-19.4], foh(14.4), 3.0)]],
)

# --- B. Wardle and Wood column ----------------------------------------------
PLANTS['ww'] = dict(
    name='Wardle and Wood (WW) column',
    T_end=900.0, ts=0.1,
    g=[[([0.126], foh(60.0), 6.0),
        ([-0.101], pmul(foh(48.0), foh(45.0)), 12.0)],
       [([0.094], foh(38.0), 8.0),
        ([-0.12], foh(35.0), 8.0)]],
)

# --- C. Vinante and Luyben column -------------------------------------------
PLANTS['vl'] = dict(
    name='Vinante and Luyben (VL) column',
    T_end=150.0, ts=0.02,
    g=[[([-2.2], foh(7.0), 1.0),
        ([1.3], foh(7.0), 0.3)],
       [([-2.8], foh(9.5), 1.8),
        ([4.3], foh(9.2), 0.35)]],
)

# --- D. Industrial Scale Polymerization process -----------------------------
PLANTS['isp'] = dict(
    name='Industrial Scale Polymerization (ISP) process',
    T_end=60.0, ts=0.01,
    g=[[([22.89], foh(4.572), 0.2),
        ([-11.64], foh(1.807), 0.4)],
       [([4.689], foh(2.174), 0.2),
        ([5.80], foh(1.801), 0.4)]],
)

# --- E. 4x4 HVAC / multi-zone process ---------------------------------------
_HVAC_K = [[-0.098, -0.036, -0.014, -0.017],
           [-0.043, -0.092, -0.011, -0.012],
           [-0.012, -0.016, -0.102, -0.033],
           [-0.013, -0.015, -0.029, -0.108]]
_HVAC_T = [[122, 149, 158, 155],
           [147, 130, 156, 157],
           [153, 151, 118, 146],
           [156, 159, 144, 128]]
_HVAC_L = [[17, 27, 32, 30],
           [25, 16, 33, 34],
           [31, 34, 16, 26],
           [32, 31, 25, 18]]

PLANTS['hvac'] = dict(
    name='4x4 HVAC multi-zone process',
    T_end=2500.0, ts=0.5,
    g=[[([_HVAC_K[i][j]], foh(_HVAC_T[i][j]), float(_HVAC_L[i][j]))
        for j in range(4)] for i in range(4)],
)

# --- F. Chen 3x3 symmetric process ------------------------------------------
_CHEN_D = ([1.2], foh(1.0), 1.0)          # diagonal:     1.2 e^-s /(s+1)
_CHEN_O = ([1.0], foh(2.0), 2.0)          # off-diagonal: e^-2s /(2s+1)

PLANTS['chen'] = dict(
    name='Chen 3x3 symmetric process',
    T_end=50.0, ts=0.02,
    g=[[(_CHEN_D if i == j else _CHEN_O) for j in range(3)] for i in range(3)],
)


def dcgain(num, den):
    return float(np.asarray(num, float)[-1] / np.asarray(den, float)[-1])

def gain_matrix(spec):
    g = spec['g']
    n = len(g)
    return np.array([[dcgain(g[i][j][0], g[i][j][1]) for j in range(n)]
                     for i in range(n)])

def rga(spec):
    K = gain_matrix(spec)
    return K * np.linalg.inv(K).T

def rga11(spec):
    return float(rga(spec)[0, 0])

def base_gains(spec):
    return [simc_pi(*fit_fopdt(*spec['g'][i][i])) for i in range(len(spec['g']))]

# ============================================================================
# plant + ideal decoupler
# ============================================================================
class MIMOPlant:
    def __init__(self, spec, ts=TS):
        self.n = len(spec['g'])
        self.g = [[TFDelay(*spec['g'][i][j], ts=ts) for j in range(self.n)]
                  for i in range(self.n)]

    def reset(self):
        for row in self.g:
            for el in row:
                el.reset()

    def step(self, u):
        return np.array([sum(self.g[i][j].step(u[j]) for j in range(self.n))
                         for i in range(self.n)])


class SimplifiedDecoupler:
    """D[i][i] = 1,  D[i][j] = -g_ij / g_ii  for i != j.

        -g_ij/g_ii = (-num_ij * den_ii) / (den_ij * num_ii),
        theta = th_ij - th_ii;  a negative theta is non-causal, so the delay
        term is dropped and the rational part retained.
    """

    def __init__(self, spec, ts=TS, warn=True):
        g = spec['g']
        self.n = n = len(g)
        self.clamped = []
        self.D = [[None] * n for _ in range(n)]

        for i in range(n):
            nii, dii, tii = g[i][i]
            for j in range(n):
                if i == j:
                    continue
                nij, dij, tij = g[i][j]
                th = tij - tii
                if th < 0:
                    self.clamped.append((f'd{i+1}{j+1}', th))
                    th = 0.0
                self.D[i][j] = TFDelay(-pmul(nij, dii), pmul(dij, nii), th, ts)

        if warn and self.clamped:
            msg = ", ".join(f"{b} theta={v:+.3g} -> 0" for b, v in self.clamped)
            warnings.warn(f"[{spec['name']}] non-causal decoupler delay dropped: {msg}")

    def reset(self):
        for row in self.D:
            for el in row:
                if el is not None:
                    el.reset()

    def step(self, v):
        u = np.empty(self.n)
        for i in range(self.n):
            u[i] = v[i] + sum(self.D[i][j].step(v[j])
                              for j in range(self.n) if j != i)
        return u

IdealDecoupler = SimplifiedDecoupler      # backwards-compatible alias


class NoDecoupler:
    def reset(self):
        pass

    def step(self, v):
        return np.asarray(v, float)


# ============================================================================
# TSK fuzzy inference
# ============================================================================
CENTERS = np.linspace(-1.0, 1.0, 7)
STEP = CENTERS[1] - CENTERS[0]

def mf_w(x, w):
    """Triangular MFs of half-width w centred on CENTERS."""
    return np.clip(1.0 - np.abs(x - CENTERS) / w, 0.0, 1.0)

def mf(x):
    """Type-1: half-width STEP -> partition of unity (sums to 1)."""
    return mf_w(x, STEP)


_B, _M, _S = 1.0, 0.5, 0.0
CP0 = np.array([
    [_B, _B, _B, _B, _B, _B, _B],
    [_M, _B, _B, _B, _B, _B, _M],
    [_S, _M, _B, _B, _B, _M, _S],
    [_S, _S, _M, _M, _M, _S, _S],
    [_S, _M, _B, _B, _B, _M, _S],
    [_M, _B, _B, _B, _B, _B, _M],
    [_B, _B, _B, _B, _B, _B, _B],
])
CI0 = 1.0 - CP0


class FuzzyPI:
    def __init__(self, Kp0, Ki0, umin=-50.0, umax=50.0, ts=TS,
                 fuzzy_type=1, delta=0.0):
        self.Kp0, self.Ki0 = Kp0, Ki0
        self.umin, self.umax, self.ts = umin, umax, ts
        self.Cp, self.Ci = CP0.copy(), CI0.copy()
        self.Ke = self.Kde = self.Ku = 1.0
        self.fuzzy_type = fuzzy_type      # 1 or 2
        self.delta = delta                # FOU width, 0 => reduces to type-1
        self.reset()

    def reset(self):
        self.v = 0.0
        self.e1 = 0.0

    def flatten(self):
        """Constant-gain mode: Kp=Kp0, Ki=Ki0 exactly."""
        self.Cp = np.full((7, 7), 0.5)
        self.Ci = np.full((7, 7), 0.5)
        return self

    def step(self, e):
        de = (e - self.e1) / self.ts
        xe = np.clip(self.Ke * e, -1, 1)
        xd = np.clip(self.Kde * de, -1, 1)

        if self.fuzzy_type == 1:
            W = np.outer(mf_w(xe, STEP), mf_w(xd, STEP))   # sums to 1
            Wn, den = W, 1.0
        else:
            d = float(np.clip(self.delta, 0.0, 0.9))
            wl, wu = STEP * (1.0 - d), STEP * (1.0 + d)
            Wl = np.outer(mf_w(xe, wl), mf_w(xd, wl))      # lower MF firing
            Wu = np.outer(mf_w(xe, wu), mf_w(xd, wu))      # upper MF firing
            Wn = Wl + Wu                                   # Nie-Tan reduction
            den = Wn.sum() + 1e-12

        dKp = float((Wn * self.Cp).sum()) / den
        dKi = float((Wn * self.Ci).sum()) / den

        Kp = self.Kp0 * (0.5 + 1.0 * dKp) * self.Ku
        Ki = self.Ki0 * (0.3 + 1.4 * dKi) * self.Ku
        dv = Kp * (e - self.e1) + Ki * self.ts * e
        self.v = float(np.clip(self.v + dv, self.umin, self.umax))
        self.e1 = e
        return self.v


# ============================================================================
# base gains: numeric FOPDT fit of the diagonal element + SIMC PI
# ============================================================================
def _timescale(den, theta):
    """Slowest time constant, for choosing fit horizon / step."""
    r = np.roots(np.asarray(den, float))
    r = r[np.abs(r) > 1e-12]
    tau_max = float(np.max(1.0 / np.abs(r))) if len(r) else 1.0
    return tau_max, 30.0 * tau_max + 10.0 * max(theta, 1e-3)


def fit_fopdt(num, den, theta):
    """Two-point (28.3%/63.2%) fit of the open-loop step response."""
    tau_max, T = _timescale(den, theta)
    ts = min(tau_max / 200.0, max(theta, 1e-3) / 20.0)
    el = TFDelay(num, den, theta, ts)
    n = int(T / ts)
    K = dcgain(num, den)
    y = np.empty(n)
    for k in range(n):
        y[k] = el.step(1.0)
    yn = y / K
    t = np.arange(n) * ts
    if yn.max() < 0.632:
        return K, tau_max, max(theta, ts)
    t28 = t[np.argmax(yn >= 0.283)]
    t63 = t[np.argmax(yn >= 0.632)]
    tau = max(1.5 * (t63 - t28), ts)
    th = max(t63 - tau, ts)
    return K, tau, th


def simc_pi(K, tau, th, tc_factor=1.0):
    tc = tc_factor * th
    Kc = tau / (K * (tc + th))
    Ti = min(tau, 4.0 * (tc + th))
    return Kc, Kc / Ti


BENCH = {
    'wb':     [(0.375, 0.375 / 8.29), (-0.075, -0.075 / 23.6)],     # BLT
    'wb_alt': [(0.208, 0.208 / 2.832), (-0.072, -0.072 / 3.802)],
}

# ============================================================================
# environment
# ============================================================================
class MIMOFuzzyEnv:
    """One action per episode.  reward = -INT sum_i |e_i| dt   (negative IAE).

    Works for any n x n plant; n is taken from the plant spec.
    Logged history layout is [t, y(n), u(n), v(n)], so for n = 2 the column
    indices are unchanged from the original 2x2 version.
    """

    T_STEP = 0
    DIVERGE_REWARD = -50.0

    def __init__(self, plant='wb', ts=None, gains=None, decouple=True,
                 T_end=None, R_step=1.0, D_time=None, D_mag=0.0,
                 tune_tables=False, warn=True, fuzzy_type=1):
        self.key = plant
        self.spec = PLANTS[plant]
        self.n_loop = n = len(self.spec['g'])
        self.ts = ts if ts is not None else self.spec['ts']

        # scalars broadcast to all loops; sequences must have length n
        self.R_STEP = np.broadcast_to(np.asarray(R_step, float), (n,)).copy()
        self.D_MAG = np.broadcast_to(np.asarray(D_mag, float), (n,)).copy()

        self.T_END = T_end if T_end is not None else self.spec['T_end']
        self.D_TIME = D_time if D_time is not None else self.T_END * 2
        self.tune_tables = tune_tables

        self.plant = MIMOPlant(self.spec, self.ts)
        self.dec = (SimplifiedDecoupler(self.spec, self.ts, warn=warn)
                    if decouple else NoDecoupler())

        g = gains if gains is not None else base_gains(self.spec)
        if len(g) != n:
            raise ValueError(f"{plant}: need {n} gain pairs, got {len(g)}")

        self.fuzzy_type = fuzzy_type
        self.n_per_loop = 4 if fuzzy_type == 2 else 3
        # two shared 7x7 rule tables (Cp, Ci) when tune_tables is on
        self.n_act = n * self.n_per_loop + (2 * 49 if tune_tables else 0)

        self.gains = g
        self.ctl = [FuzzyPI(kp, ki, ts=self.ts, fuzzy_type=fuzzy_type)
                    for kp, ki in g]

    def _apply(self, a):
        if a is None:
            return
        a = np.clip(np.asarray(a, float).ravel(), -1, 1)
        p = self.n_per_loop
        for i, c in enumerate(self.ctl):
            c.Ke = 10.0 ** (a[p * i + 0])
            c.Kde = 10.0 ** (a[p * i + 1])
            c.Ku = 10.0 ** (0.4 * a[p * i + 2])
            if self.fuzzy_type == 2:
                c.delta = 0.45 * (a[p * i + 3] + 1.0)      # -> [0, 0.9]
        if self.tune_tables:
            t = a[self.n_loop * p:].reshape(2, 7, 7)
            for c in self.ctl:
                c.Cp = np.clip(CP0 + 0.5 * t[0], 0, 1)
                c.Ci = np.clip(CI0 + 0.5 * t[1], 0, 1)

    def rollout(self, action=None, log=False, seed=None):
        self._apply(action)
        self.plant.reset()
        self.dec.reset()
        for c in self.ctl:
            c.reset()

        n_loop = self.n_loop
        n_step = int(round(self.T_END / self.ts))
        y = np.zeros(n_loop)
        zero = np.zeros(n_loop)
        iae = 0.0
        hist = []

        for k in range(n_step):
            t = k * self.ts
            r = self.R_STEP if t >= self.T_STEP else zero
            e = r - y
            v = np.array([self.ctl[i].step(e[i]) for i in range(n_loop)])
            u = self.dec.step(v)
            if t >= self.D_TIME:
                u = u + self.D_MAG
            y = self.plant.step(u)

            if not np.all(np.isfinite(y)) or np.abs(y).max() > 1e4:
                h = np.array(hist) if hist else np.empty((0, 1 + 3 * n_loop))
                return (self.DIVERGE_REWARD, h) if log else self.DIVERGE_REWARD

            iae += np.abs(e).sum() * self.ts
            if log:
                hist.append(np.concatenate(([t], y, u, v)))

        h = np.array(hist) if hist else np.empty((0, 1 + 3 * n_loop))
        r_ = -iae
        return (r_, h) if log else r_

    def metrics(self, action=None):
        """Per-loop and total ISE / IAE from a logged rollout."""
        _, h = self.rollout(action, log=True)
        n = self.n_loop
        if len(h) == 0:
            nan = np.full(n, np.nan)
            return dict(ise=nan, iae=nan, ise_tot=np.inf, iae_tot=np.inf, hist=h)
        t = h[:, 0]
        R = np.where(t[:, None] >= self.T_STEP, self.R_STEP[None, :], 0.0)
        e = R - h[:, 1:1 + n]
        ise = (e ** 2).sum(0) * self.ts
        iae = np.abs(e).sum(0) * self.ts
        return dict(ise=ise, iae=iae, ise_tot=ise.sum(), iae_tot=iae.sum(), hist=h)


if __name__ == "__main__":
    for key in ('wb', 'ww', 'vl', 'isp'):
        spec = PLANTS[key]
        dec = IdealDecoupler(spec, spec['ts'], warn=False)
        cl = "; ".join(f"{b} {v:+.3g}->0" for b, v in dec.clamped) or "none"
        print(f"\n=== {spec['name']} ===")
        print(f"  RGA l11 = {rga11(spec):7.3f}   ts = {spec['ts']}   "
              f"T_end = {spec['T_end']}   clamped: {cl}")
        for i in range(2):
            K, tau, th = fit_fopdt(*spec['g'][i][i])
            kc, ki = simc_pi(K, tau, th)
            print(f"  loop {i+1}: fit K={K:9.4f} tau={tau:8.3f} theta={th:7.3f}"
                  f"  -> Kc={kc:9.4f} Ki={ki:9.5f}")
        env = MIMOFuzzyEnv(plant=key, warn=False)
        m = env.metrics(np.zeros(6))
        print(f"  default action: ISE {m['ise_tot']:9.3f}  IAE {m['iae_tot']:9.3f}"
              f"   (ISE y1 {m['ise'][0]:.3f} / y2 {m['ise'][1]:.3f})")