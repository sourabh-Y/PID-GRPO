"""
Baseline designs that follow each paper's own procedure (no GRPO).

Sahin & Ulu (2026), design procedure of Sec. 4:
  1. PID gains from a classical design          -> the shared IMC-PID
  2. SI-IT2-FPID with constant FOU: choose m0   -> 1-D grid per loop, k1 = 0
  3. OTM: k2 = m0, choose k1 in [0.02, 0.98]    -> 1-D grid per loop
  The paper chooses m0 and k1 by trial; here a grid over the paper's bounds
  replaces the trial, objective IAE on the nominal plant, loops one at a time
  (coordinate sweeps), because the papers are SISO.

Kluska & Zabinski (2020), Steps 1-5:
  1-2. linear PID and certified sector          -> KluskaEnv (baselines.py)
  3.   reference model: each FC-2 consequent takes one of the paper's three
       candidates  Q = beta H, (beta+K) H / 2, (beta+K) H;
       exhaustive search, 3^8 = 6561 step responses per loop on the nominal
       plant; selection "no overshoot, shortest settling time" (criterion
       'paper') or lowest IAE (criterion 'iae').  Loops one at a time.
  4.   initial consequents = linear PID          -> q_init='pid' at evaluation
  5.   adaptation gain                           -> kluska_select_gamma
"""
import itertools
import numpy as np

from baselines import make_env, SahinUluEnv, SU_LO, SU_HI

_CACHE = {}


def _get_env(kw):
    key = repr(sorted(kw.items()))
    env = _CACHE.get(key)
    if env is None:
        env = make_env(**kw)
        _CACHE[key] = env
    return env


def _map(pool, fn, args):
    if pool is None:
        return [fn(a) for a in args]
    return pool.map(fn, args, chunksize=max(1, len(args) // (8 * pool._processes)))


def _inv_pw(v, nom, lo, hi):
    """Inverse of baselines._pw: value -> action in [-1, 1]."""
    v = np.asarray(v, float)
    return np.where(v >= nom, (v - nom) / (hi - nom), (v - nom) / (nom - lo))


# ============================================================================
# Sahin & Ulu
# ============================================================================
def su_eval(args):
    """args = (env kwargs, k1 per loop, k2 per loop) -> reward (-IAE)."""
    kw, k1s, k2s = args
    env = _get_env(kw)
    for c, k1, k2 in zip(env.ctl, k1s, k2s):
        c.k1, c.k2 = float(k1), float(k2)
    return env.rollout(None)


def _coord_search(pool, kw, fixed, var, grid, which, sweeps, verbose, tag):
    """Coordinate search on one parameter (k1 or k2) per loop."""
    n = len(var)
    var = list(var)

    def args_for(v):
        return (kw, v, fixed) if which == 'k1' else (kw, fixed, v)

    best = su_eval(args_for(var))
    for s in range(sweeps):
        changed = False
        for i in range(n):
            cands = [var[:i] + [g] + var[i + 1:] for g in grid]
            rs = np.array(_map(pool, su_eval, [args_for(v) for v in cands]))
            j = int(np.argmax(rs))
            if rs[j] > best + 1e-12:
                best, var, changed = float(rs[j]), cands[j], True
            if verbose:
                print(f"  {tag} sweep {s + 1} loop {i + 1}: {which} = "
                      f"{var[i]:.3f}   IAE {-best:.3f}")
        if not changed:
            break
    return var, best


def design_sahin_ulu(env_kw, pool=None, grid=np.linspace(0.02, 0.98, 25),
                     sweeps=2, verbose=True):
    kw = {**env_kw, 'method': 'sahin_ulu', 'tune_pid': False}
    env = _get_env(kw)
    n = env.n_loop

    # step 2: constant FOU (k1 = 0, alpha = m0)
    m0, r_const = _coord_search(pool, kw, fixed=[0.0] * n,
                                var=[SahinUluEnv.K2_NOM] * n, grid=grid,
                                which='k2', sweeps=sweeps, verbose=verbose,
                                tag='constant FOU')
    # step 3: OTM with k2 = m0, choose k1
    k1, r_otm = _coord_search(pool, kw, fixed=m0, var=[SU_LO] * n, grid=grid,
                              which='k1', sweeps=sweeps, verbose=verbose,
                              tag='OTM')

    a = np.concatenate([[_inv_pw(k1[i], SahinUluEnv.K1_NOM, SU_LO, SU_HI),
                         _inv_pw(m0[i], SahinUluEnv.K2_NOM, SU_LO, SU_HI)]
                        for i in range(n)])
    log = dict(m0=np.array(m0), k1=np.array(k1),
               iae_const_fou=-r_const, iae_otm=-r_otm,
               iae_pid_start=-su_eval((kw, [0.0] * n, [SahinUluEnv.K2_NOM] * n)))
    return a, log


# ============================================================================
# Kluska & Zabinski
# ============================================================================
def kz_eval(args):
    """args = (env kwargs, lambda per loop) -> (reward, overshoot, settling)."""
    kw, lams = args
    env = _get_env(kw)
    for c, l in zip(env.ctl, lams):
        c.set_lam(l)
        c.adaptive = False
    r, h = env.rollout(None, log=True)
    n = env.n_loop
    if len(h) < int(round(env.T_END / env.ts)):
        return -np.inf, [np.inf] * n, [np.inf] * n
    t = h[:, 0]
    over, settle = [], []
    for i in range(n):
        R = env.R_STEP[i] if abs(env.R_STEP[i]) > 1e-12 else 1.0
        y = h[:, 1 + i] / R
        over.append(max(0.0, float(y.max()) - 1.0))
        out = np.flatnonzero(np.abs(y - 1.0) > 0.02)            # 2% band
        settle.append(float(t[out[-1]] + env.ts) if len(out) else 0.0)
    return r, over, settle


def design_kluska(env_kw, pool=None, criterion='paper', os_tol=0.01,
                  verbose=True):
    kw = {**env_kw, 'method': 'kluska', 'adaptive': False}
    env = _get_env(kw)
    n = env.n_loop

    lams = [np.full(len(c.nz), c.lam0) for c in env.ctl]       # Step 4 start
    cand = [np.array([0.0, float(np.clip((c.K - c.beta) / (2.0 * c.K), 0, 1)), 1.0])
            for c in env.ctl]
    log = {}

    for i in range(n):
        combos = np.array(list(itertools.product(cand[i], repeat=len(lams[i]))))
        res = _map(pool, kz_eval,
                   [(kw, lams[:i] + [cb] + lams[i + 1:]) for cb in combos])
        r = np.array([x[0] for x in res])
        ov = np.array([x[1][i] for x in res])
        st = np.array([x[2][i] for x in res])

        if criterion == 'paper':
            ok = np.flatnonzero(np.isfinite(r) & (ov <= os_tol))
            if len(ok):
                j = ok[np.lexsort((-r[ok], st[ok]))[0]]       # min settling, then IAE
            else:
                fin = np.flatnonzero(np.isfinite(r))
                j = fin[np.lexsort((-r[fin], ov[fin]))[0]]    # min overshoot
        else:
            j = int(np.argmax(r))

        lams[i] = combos[j]
        log[f'loop{i + 1}_overshoot'] = ov[j]
        log[f'loop{i + 1}_settling'] = st[j]
        log[f'loop{i + 1}_n_no_overshoot'] = int((np.isfinite(r) & (ov <= os_tol)).sum())
        if verbose:
            print(f"  loop {i + 1}: {len(combos)} candidates, "
                  f"{log[f'loop{i + 1}_n_no_overshoot']} without overshoot; chosen "
                  f"overshoot {100 * ov[j]:.2f}%  settling {st[j]:.2f}  "
                  f"IAE {-r[j]:.3f}")

    a = np.concatenate([_inv_pw(l, c.lam0, 0.0, 1.0)
                        for l, c in zip(lams, env.ctl)])
    log['lambda'] = np.concatenate(lams)
    return a, log
