from mimo_fuzzy_env import MIMOPlant
from baselines import make_env, sampled_spec

_ENV = None
_NOM_PLANT = None
_KEY = 'nominal'


def init_worker(env_kw):
    global _ENV, _NOM_PLANT, _KEY
    _ENV = make_env(**env_kw)          # one env per worker process
    _NOM_PLANT = _ENV.plant
    _KEY = 'nominal'


def rollout(a):
    return _ENV.rollout(a)


def rollout_on(args):
    """args = (action, delta, plant_seed); plant_seed=None means nominal."""
    global _KEY
    a, delta, plant_seed = args
    key = 'nominal' if plant_seed is None else plant_seed
    if key != _KEY:
        _ENV.plant = (_NOM_PLANT if plant_seed is None else
                      MIMOPlant(sampled_spec(_ENV.spec, delta, plant_seed), _ENV.ts))
        _KEY = key
    return _ENV.rollout(a)
