"""Budget-matched memetic search and baselines; test arrays never enter this API."""
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from .config import METHODS
from .physics import (FULL_LIBRARY, RESTRICTED_LIBRARY, HARD_WIDTHS,
                      intensity, calibrate, expected_mse)
from .data import roi


def initial_parameters(config, rng, conventional=False):
    c = config
    p = {"gaps": rng.normal(0, .2, c.stations+1),
         "focal": rng.normal(-1, .4, c.stations),
         "width": rng.normal(0, .5, c.stations),
         "phase": rng.normal(.55, .1, c.stations),
         "masks": rng.normal(0, .05, (c.stations, c.mask_grid, c.mask_grid))}
    if conventional:
        fraction = np.clip((c.length_um/4-c.focal_min_um)/(c.focal_max_um-c.focal_min_um), .001, .999)
        p["gaps"][:] = 0
        p["focal"][:] = np.log(fraction/(1-fraction))
        p["width"][:] = -3
        p["phase"][:] = np.arctanh(.5)  # pi/2 phase disk
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in p.items()}


def repair(architecture, config, rng):
    a = np.array(architecture, dtype=np.int32, copy=True)
    active = np.flatnonzero(a != 2)
    if len(active) > config.max_components:
        a[rng.choice(active, len(active)-config.max_components, replace=False)] = 2
    return a


def validate_method(config, method):
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}; choose {METHODS}")
    if method in ("fixed", "freeform") and config.max_components < config.stations:
        raise ValueError("Fixed/freeform baselines require max_components == stations")
    if method == "fixed" and config.stations != 3:
        raise ValueError("The fixed lens/phase-disk/lens baseline requires three stations")


def search(config, train, validation, method, seed, progress=None):
    validate_method(config, method)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Search seed must be a uint32 integer")
    c = config
    for name, images in (("train", train), ("validation", validation)):
        if images.ndim != 3 or images.shape[1:] != (c.size, c.size) or not len(images):
            raise ValueError(f"Invalid {name} shape")
        if not np.isfinite(images).all():
            raise ValueError(f"Nonfinite {name} data")
    train, validation = jnp.asarray(train), jnp.asarray(validation)
    rng = np.random.default_rng(seed)
    library = RESTRICTED_LIBRARY if method == "restricted" else FULL_LIBRARY
    fixed = np.array([1, 12, 1], dtype=np.int32)
    noise_aware = method != "no_noise"
    optimizer = optax.chain(optax.clip_by_global_norm(10.), optax.adam(c.learning_rate))

    def signals(p, a, images):
        return jax.vmap(lambda dz: jax.vmap(lambda phase:
            intensity(p, a, phase, c, dz))(images))(jnp.array(c.train_defocus_um))

    def objective(p, a):
        s = signals(p, a, train)
        calibration = calibrate(s, train, c, noise_aware)
        risk = jnp.mean(expected_mse(s, train, calibration, c, noise_aware=noise_aware))
        return risk + c.component_penalty*jnp.count_nonzero(a != 2)

    @jax.jit
    def train_candidate(p, a):
        # Reset Adam at each candidate/generation for every method, including
        # elites. Warm-started optical weights are retained; moments are not.
        def step(carry, _):
            p, state, rejected = carry
            value, grads = jax.value_and_grad(objective)(p, a)
            # Hard sampled edges use evolutionary width changes only.
            hard = jnp.isin(a, jnp.array(HARD_WIDTHS))
            grads = dict(grads, width=jnp.where(hard, 0., grads["width"]))
            updates, next_state = optimizer.update(grads, state, p)
            next_p = jax.tree_util.tree_map(lambda x: jnp.clip(x, -12., 12.),
                                            optax.apply_updates(p, updates))
            finite = jnp.isfinite(value)
            for leaf in jax.tree_util.tree_leaves((grads, next_p, next_state)):
                finite = finite & jnp.all(jnp.isfinite(leaf))
            return jax.lax.cond(finite,
                lambda: (next_p, next_state, rejected),
                lambda: (p, state, rejected+1)), None
        (p, _, rejected), _ = jax.lax.scan(step, (p, optimizer.init(p), jnp.array(0)),
                                          None, length=c.gradient_steps)
        return p, rejected

    @jax.jit
    def score(p, a):
        calibration = calibrate(signals(p, a, train), train, c, noise_aware)
        mse = jnp.mean(expected_mse(signals(p, a, validation), validation,
                                   calibration, c, noise_aware=noise_aware))
        return mse + c.component_penalty*jnp.count_nonzero(a != 2), mse, calibration

    if method in ("constant", "defocus"):
        start = time.perf_counter()
        p = initial_parameters(c, rng)
        a = jnp.full(c.stations, 2, dtype=jnp.int32)
        if method == "constant":
            calibration = {"gain": jnp.array(0.), "offset": jnp.mean(train[:, roi(c.size)]),
                           "inverted": jnp.array(False)}
        else:
            calibration = calibrate(signals(p, a, train), train, c)
        mse = float(jnp.mean(expected_mse(signals(p, a, validation), validation, calibration, c)))
        return {"architecture": np.asarray(a),
                "parameters": jax.tree_util.tree_map(np.asarray, p),
                "calibration": {k: np.asarray(v).item() for k,v in calibration.items()},
                "validation_objective": mse, "validation_mse_rad2": mse, "generation": 0,
                "history": [], "method": method, "seed": seed, "rejected_gradient_steps": 0,
                "candidate_evaluations": 1, "gradient_steps": 0,
                "wall_seconds_including_compilation": time.perf_counter()-start}

    def new_candidate(i):
        if method == "fixed":
            a = fixed.copy()
        elif method == "freeform":
            a = np.full(c.stations, 13, dtype=np.int32)
        else:
            a = repair(rng.choice(library, c.stations), c, rng)
        p = initial_parameters(c, rng, conventional=(method == "fixed" and i == 0))
        return a, p

    population = [new_candidate(i) for i in range(c.population)]
    best, history, rejected_steps = None, [], 0
    start = time.perf_counter()
    for generation in range(c.generations):
        scored = []
        for a, p in population:
            p, rejected = train_candidate(p, jnp.asarray(a))
            rejected_steps += int(rejected)
            objective_value, mse, calibration = score(p, jnp.asarray(a))
            value = float(objective_value)
            if not np.isfinite(value):
                value = float("inf")
            scored.append((value, a, p))
            if np.isfinite(value) and (best is None or value < best["validation_objective"]):
                best = {"architecture": a.copy(),
                        "parameters": jax.tree_util.tree_map(lambda x: np.array(x, copy=True), p),
                        "calibration": {k: np.asarray(v).item() for k, v in calibration.items()},
                        "validation_objective": value, "validation_mse_rad2": float(mse),
                        "generation": generation+1}
        if best is None:
            raise FloatingPointError("All candidates are nonfinite; check configuration and sampling")
        history.append({"generation": generation+1,
                        "best_validation_objective": best["validation_objective"],
                        "best_validation_mse_rad2": best["validation_mse_rad2"],
                        "candidate_evaluations": (generation+1)*c.population,
                        "gradient_steps": (generation+1)*c.population*c.gradient_steps,
                        "wall_seconds_including_compilation": time.perf_counter()-start})
        if progress:
            progress(history[-1])
        if generation == c.generations-1:
            break
        if method == "random":
            population = [new_candidate(i) for i in range(c.population)]
            continue
        scored.sort(key=lambda x: x[0])
        elite = [x for x in scored[:max(1, c.population//4)] if np.isfinite(x[0])]
        population = [(a.copy(), p) for _, a, p in elite]
        while len(population) < c.population:
            _, a, p = elite[int(rng.integers(len(elite)))]
            a = a.copy()
            if method not in ("fixed", "freeform") and rng.random() < c.mutation_probability:
                index = rng.integers(c.stations)
                a[index] = rng.choice([op for op in library if op != a[index]])
                a = repair(a, c, rng)
            child = {k: jnp.asarray(np.clip(np.asarray(v) + rng.normal(0, .15, v.shape), -12, 12),
                                    dtype=jnp.float32) for k, v in p.items()}
            population.append((a, child))
    best.update(history=history, method=method, seed=seed, rejected_gradient_steps=rejected_steps,
                candidate_evaluations=c.population*c.generations,
                gradient_steps=c.population*c.generations*c.gradient_steps,
                wall_seconds_including_compilation=time.perf_counter()-start)
    return best
