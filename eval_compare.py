"""
IAE / ISE of every method over seeds, on the nominal, +30% and -30% plants.
Run in the folder that holds the a_fis_*.npy files:   python eval_compare.py
Writes every number to results_all.csv as well.
"""
import os
import csv
import numpy as np

from mimo_fuzzy_env import MIMOPlant, PLANTS
from baselines import make_env, perturb

PLANT_KEYS = ('vl', 'wb', 'ww', 'isp', 'hvac')
SEEDS = (0, 1, 2, 3, 4)
SHOW_SEEDS = False            # True: also print one row per seed
TR, TAUC_F = 'eiasc', 1.0
TAU_F_OF = {'hvac': 2.0, 'chen': 0.1}
PERTS = [('nominal', 0.0), ('+30%', 0.30), ('-30%', -0.30)]

# label: (file tag, env kwargs, meta key with Kluska's gamma factor)
MODES = {
    'Type-1':            ('t1',       dict(fuzzy_type=1), None),
    'Fixed FOU':         ('it2',      dict(fuzzy_type=2), None),
    'Tuned FOU':         ('fou',      dict(fuzzy_type=2, tune_fou=True), None),
    'Type-1 (DR)':       ('t1_dr15',  dict(fuzzy_type=1), None),
    'Fixed FOU (DR)':    ('it2_dr15', dict(fuzzy_type=2), None),
    'Tuned FOU (DR)':    ('fou_dr15', dict(fuzzy_type=2, tune_fou=True), None),
    # paper designs: one per plant, no seeds (files a_fis_{plant}_{tag}.npy)
    'Sahin-Ulu':         ('su_paper', dict(method='sahin_ulu', tune_pid=False), None),
    'Kluska':            ('kz_paper', dict(method='kluska', adaptive=True,
                                           q_init='pid'), 'gamma_pid'),
    'Type-1 (δ=5%)':  ('t1_dr5',   dict(fuzzy_type=1), None),
    'IT2 (δ=5%)':     ('fou_dr5',  dict(fuzzy_type=2, tune_fou=True), None),
    'Type-1 (δ=30%)': ('t1_dr30',  dict(fuzzy_type=1), None),
    'IT2 (δ=30%)':    ('fou_dr30', dict(fuzzy_type=2, tune_fou=True), None),                                       
}


def tau_f_for(plant):
    return max(TAU_F_OF.get(plant, 0.01), 4.0 * PLANTS[plant]['ts'])


def base_kw(plant):
    return dict(plant=plant, R_step=1.0, warn=False, tr_method=TR,
                tauC_factor=TAUC_F, tau_f=tau_f_for(plant), antiwindup='clamp')


def evaluate(env, a, p, nominal_plant):
    env.plant = (nominal_plant if p == 0.0
                 else MIMOPlant(perturb(env.spec, p), env.ts))
    m = env.metrics(a)
    if len(m['hist']) < int(round(env.T_END / env.ts)):
        return np.nan, np.nan
    return m['iae_tot'], m['ise_tot']


def cell(x):
    return f"{'UNSTABLE' if np.isnan(x) else f'{x:.2f}':>17s}"


def summary(v):
    v = np.asarray(v, float)
    if np.isnan(v).any():
        return f"{f'UNSTABLE ({int(np.isnan(v).sum())})':>17s}"
    if len(v) < 2:
        return f"{f'{v.mean():.2f}':>17s}"
    return f"{f'{v.mean():.2f} ± {v.std(ddof=1):.2f}':>17s}"


rows_csv = []
header = f"{'method':<19s}{'seeds':<7s}" + "".join(
    f"{lbl + ' IAE':>17s}{lbl + ' ISE':>17s}" for lbl, _ in PERTS)

for plant in PLANT_KEYS:
    print(f"\n{PLANTS[plant]['name']}")
    print(header)
    print('-' * len(header))

    for label, (tag, extra, gkey) in MODES.items():
        env, rows = None, []
        seeds = (None,) if tag.endswith('_paper') else SEEDS
        for seed in seeds:
            fname = (f"a_fis_{plant}_{tag}" if seed is None
                     else f"a_fis_{plant}_{tag}_seed{seed}")
            if not os.path.exists(fname + ".npy"):
                continue
            if env is None:
                env = make_env(**base_kw(plant), **extra)
                nominal_plant = env.plant
            a = np.load(fname + ".npy")
            if len(a) != env.n_act:
                print(f"{label:<19s}seed {seed}: action has {len(a)} values, "
                      f"needs {env.n_act}")
                continue
            if gkey is not None:
                meta = np.load(fname + "_meta.npz", allow_pickle=True)
                env.set_gamma_factor(float(meta[gkey]))

            row = []
            for lbl, p in PERTS:
                iae, ise = evaluate(env, a, p, nominal_plant)
                row += [iae, ise]
                rows_csv.append([plant, label, '' if seed is None else seed,
                                 lbl, iae, ise])
            rows.append(row)
            if SHOW_SEEDS and seed is not None:
                print(f"{label:<19s}{seed:<7d}" + "".join(cell(x) for x in row))

        if not rows:
            print(f"{label:<19s}no files")
            continue
        rows = np.array(rows, float)
        print(f"{label:<19s}{len(rows):<7d}"
              + "".join(summary(rows[:, j]) for j in range(rows.shape[1])))

with open("results_all.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["plant", "method", "seed", "condition", "IAE", "ISE"])
    w.writerows(rows_csv)
print("\nall values -> results_all.csv")
