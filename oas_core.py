"""Testable optics and memetic search, independent of Streamlit and CIFAR downloads.

Lengths are in micrometres. The monochromatic scalar model is paraxial.
The phase task measures weak, relative phase contrast, not absolute phase.
"""

from dataclasses import dataclass

import chromatix.functional as cx
from chromatix import crop, pad
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
import optax

AMPLITUDE = "5.1. Amplitude Imaging"
PHASE = "5.2. Phase Imaging"
OPS = (
    "Propagate", "ThinLens", "Identity", "CircularPupil", "SquarePupil",
    "GaussianPupil", "SuperGaussianPupil", "RectangularPupil", "TukeyPupil",
    "Axicon", "SawtoothGrating", "SinusoidGrating",
)


def normalized_mse(prediction, target):
    """Relative MSE with a floor below the physical intensities used here."""
    if prediction.shape != target.shape:
        raise ValueError(f"Image shapes differ: {prediction.shape} != {target.shape}")
    scale = jnp.maximum(jnp.mean(target**2), 1e-20)
    return jnp.mean((prediction - target) ** 2) / scale


def standardized(image, epsilon=1e-6):
    """Finite derivative at zero variance; epsilon is INSIDE the square root."""
    centered = image - jnp.mean(image)
    return centered / jnp.sqrt(jnp.mean(centered**2) + epsilon**2)


def generate_target_image(shape, dx, spectrum, f1, f2, obj, task=AMPLITUDE):
    """Ideal 4f target sampled on the *same sensor grid* as the candidates.

    For positive f1/f2, u_out(r) = u_in(-r/M)/M up to a global phase, M=f2/f1.
    Evaluate that amplitude map directly, avoiding the changed sampling and
    periodic boundary wrap of two discrete optical FFTs. Objects are zero
    outside their support. Linear interpolation is not pixel integration.
    """
    if task == PHASE:
        return jnp.asarray(obj)
    if task != AMPLITUDE:
        raise ValueError(f"Unknown task: {task}")
    if f1 <= 0 or f2 <= 0:
        raise ValueError("Target focal lengths must be positive")
    if tuple(obj.shape) != tuple(shape):
        raise ValueError("Object must match the simulation shape")
    coords = jnp.meshgrid(*[
        -(jnp.arange(size) - size / 2) * (f1 / f2) + size / 2
        for size in shape
    ], indexing="ij")
    amplitude = map_coordinates(jnp.asarray(obj), coords, order=1, mode="constant", cval=0.0)
    return amplitude**2 * (f1 / f2)**2 / (np.prod(shape) * dx**2)


def project_parameters(params, shape, dx, spectrum, pad_width=0):
    """Positive, bounded parameters; returned values are the simulated values.

    The lens lower bound limits its own adjacent phase increments on the
    working grid. This alone does NOT guarantee a resolved propagated field.
    """
    z, f, w = params
    extent = max(shape) * dx + 2 * pad_width * dx
    min_f = max(1.0, 1.05 * extent * dx / spectrum)
    return (
        jnp.clip(z, 1.0, 40000.0),
        jnp.clip(f, min_f, max(40000.0, 4 * min_f)),
        jnp.clip(w, 2 * dx, extent),
    )


def input_field(shape, dx, spectrum, obj, task, pad_width=0, phase_scale=0.5):
    if tuple(obj.shape) != tuple(shape):
        raise ValueError("Object must match the simulation shape")
    field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
    if task == AMPLITUDE:
        field = cx.amplitude_change(field, obj)
    elif task == PHASE:
        # A global phase cannot be detected by an intensity-only linear system.
        # Restrict the task to weak relative phase, avoiding a 0/2pi ambiguity.
        field = cx.phase_change(field, phase_scale * (obj - jnp.mean(obj)))
    else:
        raise ValueError(f"Unknown task: {task}")
    return pad(field, pad_width)


def apply_operation(field, index, z, f, w, dx, spectrum):
    """Execute one operation on the existing working grid.

    Hard pupils remain exact hard pupils. Their widths are evolved, since
    sampled Boolean masks have zero width derivative almost everywhere.
    Gratings use one-wave (sawtooth) / pi-radian (sinusoid) phase depth instead
    of tying thickness to period, which severely aliased the original masks.
    """
    period = jnp.maximum(w / 2, 8 * dx)
    angle = (w / (w + 100.0)) * jnp.minimum(0.1, spectrum / (4 * 0.5 * dx))
    return jax.lax.switch(index, (
        lambda fld: cx.transfer_propagate(fld, z=z, n=1.0, pad_width=0, mode="same"),
        lambda fld: cx.thin_lens(fld, f=f, n=1.0),
        lambda fld: fld,
        lambda fld: cx.circular_pupil(fld, w=w),
        lambda fld: cx.square_pupil(fld, w=w),
        lambda fld: cx.gaussian_pupil(fld, w=w),
        lambda fld: cx.super_gaussian_pupil(fld, w=w, n=16.0),
        lambda fld: cx.rectangular_pupil(fld, h=w / 2, w=w),
        lambda fld: cx.tukey_pupil(fld, w=w),
        lambda fld: cx.axicon(fld, n_axicon=1.5, slope_angle=angle),
        lambda fld: cx.sawtooth_grating(fld, n_grating=1.5, period=period,
                                      thickness=spectrum / 0.5),
        lambda fld: cx.sinusoid_grating(fld, n_grating=1.5, period=period,
                                      thickness=spectrum),
    ), field)


def simulate_discrete_architecture(z, f, w, arch_indices, shape, dx, spectrum,
                                   obj, pad_width, task=AMPLITUDE):
    field = input_field(shape, dx, spectrum, obj, task, pad_width)
    # Pad once and crop at the sensor. Cropping after every free-space segment
    # inserted unintended stops and made P(z1)P(z2) differ from P(z1+z2).
    def step(fld, values):
        index, zi, fi, wi = values
        return apply_operation(fld, index, zi, fi, wi, dx, spectrum), None
    field, _ = jax.lax.scan(step, field, (jnp.asarray(arch_indices), z, f, w))
    return crop(field, pad_width).intensity


def loss_fn(z, f, w, arch_indices, shape, dx, spectrum, obj, target_I,
            pad_width, task=AMPLITUDE):
    output = simulate_discrete_architecture(
        z, f, w, arch_indices, shape, dx, spectrum, obj, pad_width, task
    )
    if output.shape != target_I.shape:
        raise ValueError("Prediction and target must use the same sensor grid")
    if task == AMPLITUDE:
        return normalized_mse(output, target_I)
    # Normalize the intensity units first so numerical roundoff in a constant
    # field is not amplified to significant contrast.
    incident_intensity = 1.0 / (np.prod(shape) * dx**2)
    return jnp.mean((standardized(output / incident_intensity)
                     - standardized(target_I)) ** 2)


def describe_architecture(arch, params, dx=5.0, spectrum=0.532):
    """Export actual physical quantities, including phase-element units."""
    z, f, w = (np.asarray(x) for x in params)
    result = []
    for i, index in enumerate(np.asarray(arch)):
        op = OPS[int(index)]
        values = {}
        if op == "Propagate":
            values = {"z_um": float(z[i])}
        elif op == "ThinLens":
            values = {"f_um": float(f[i])}
        elif op == "Axicon":
            values = {"slope_angle_rad": float(w[i] / (w[i] + 100.0)
                       * min(0.1, spectrum / (4 * 0.5 * dx)))}
        elif "Grating" in op:
            values = {"period_um": float(max(w[i] / 2, 8 * dx)),
                      "thickness_um": float(spectrum / 0.5 if op == "SawtoothGrating" else spectrum)}
        elif op != "Identity":
            values = {"w_um": float(w[i])}
            if op == "RectangularPupil":
                values["h_um"] = float(w[i] / 2)
        result.append({"operation": op, **values})
    return result


def sampling_warnings(arch, params, shape, dx, spectrum, pad_width):
    """Conservative adjacent-phase checks, not a substitute for convergence."""
    z, f, _ = (np.asarray(x) for x in params)
    arch = np.asarray(arch)
    sizes = np.asarray(shape) + 2 * pad_width
    messages = []
    if np.any(f[arch == 1] < max(sizes) * dx**2 / spectrum):
        messages.append("At least one thin lens is undersampled on the working grid.")
    if np.any(z[arch == 0] > min(sizes) * dx**2 / spectrum):
        messages.append("At least one Fresnel transfer kernel is undersampled; increase padding.")
    return messages


@dataclass(frozen=True)
class SearchConfig:
    shape: tuple[int, int] = (64, 64)
    dx: float = 5.0
    spectrum: float = 0.532
    pad_width: int = 64
    f1: float = 4000.0
    f2: float = 4000.0
    task: str = AMPLITUDE
    num_blocks: int = 7
    pop_size: int = 15
    gd_steps: int = 5
    generations: int = 60
    seed: int = 42

    def __post_init__(self):
        if self.task not in (AMPLITUDE, PHASE):
            raise ValueError("Unknown imaging task")
        integers = (self.num_blocks, self.pop_size, self.gd_steps, self.generations, *self.shape)
        if any(not isinstance(x, (int, np.integer)) or x < 1 for x in integers):
            raise ValueError("Search counts and image dimensions must be positive integers")
        if not isinstance(self.pad_width, (int, np.integer)) or self.pad_width < 0:
            raise ValueError("Padding must be a nonnegative integer")
        if any(not np.isfinite(x) or x <= 0 for x in (self.dx, self.spectrum, self.f1, self.f2)):
            raise ValueError("Pixel size, wavelength and focal lengths must be finite and positive")
        if not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("Seed must be a nonnegative integer")


@dataclass
class SearchResult:
    architecture: np.ndarray
    parameters: tuple
    validation_loss: float
    history: list[float]
    initial_architecture: np.ndarray
    initial_parameters: tuple


def best_candidate(losses):
    losses = np.asarray(losses)
    valid = np.isfinite(losses)
    if not valid.any():
        raise FloatingPointError("All candidate losses are non-finite; check optics and sampling")
    return int(np.argmin(np.where(valid, losses, np.inf)))


def _check_batch(batch, config, name):
    batch = np.asarray(batch, dtype=np.float32)
    if batch.ndim != 3 or batch.shape[0] == 0 or batch.shape[1:] != config.shape:
        raise ValueError(f"{name} must have shape (N, {config.shape[0]}, {config.shape[1]}) with N > 0")
    if not np.isfinite(batch).all() or np.any((batch < 0) | (batch > 1)):
        raise ValueError(f"{name} must contain finite object values in [0, 1]")
    return jnp.asarray(batch)


def run_search(config, train_images, validation_images, progress=None):
    """Train on random training images; select on a fixed validation batch.

    Save a copy of the best evaluated candidate BEFORE breeding. Preserve an
    unchanged elite and its optimizer state; reset Adam for perturbed children.
    The untouched test split must be evaluated separately after this function.
    """
    train = _check_batch(train_images, config, "Training images")
    validation = _check_batch(validation_images, config, "Validation images")
    c = config
    rng = np.random.default_rng(c.seed)
    projection = lambda p: project_parameters(p, c.shape, c.dx, c.spectrum, c.pad_width)
    arch = jnp.asarray(rng.integers(len(OPS), size=(c.pop_size, c.num_blocks)), dtype=jnp.int32)
    low_f = float(projection((jnp.array(1.), jnp.array(1.), jnp.array(1.)))[1])
    params = projection(tuple(jnp.asarray(x, dtype=jnp.float32) for x in (
        rng.uniform(1000, 8000, (c.pop_size, c.num_blocks)),
        rng.uniform(low_f, max(low_f + 1, 2 * low_f), (c.pop_size, c.num_blocks)),
        rng.uniform(50, 300, (c.pop_size, c.num_blocks)),
    )))
    initial_arch = np.array(arch[0], copy=True)
    initial_params = tuple(np.array(p[0], copy=True) for p in params)
    # Widths/periods have a much smaller scale than distances and focal lengths.
    optimizer = optax.multi_transform(
        {"distance": optax.adam(50.), "width": optax.adam(2.)},
        ("distance", "distance", "width"),
    )
    state = jax.vmap(optimizer.init)(params)

    def target(image):
        return generate_target_image(c.shape, c.dx, c.spectrum, c.f1, c.f2, image, c.task)

    train_targets = jax.jit(jax.vmap(target))(train)
    val_targets = jax.jit(jax.vmap(target))(validation)

    def objective(p, a, image, truth):
        return loss_fn(*p, a, c.shape, c.dx, c.spectrum, image, truth, c.pad_width, c.task)

    def step(p, state_i, a, image, truth):
        value, grads = jax.value_and_grad(objective)(p, a, image, truth)
        updates, new_state = optimizer.update(grads, state_i, p)
        new_p = projection(optax.apply_updates(p, updates))
        finite = jnp.isfinite(value)
        for leaf in jax.tree_util.tree_leaves((grads, new_p, new_state)):
            finite = finite & jnp.all(jnp.isfinite(leaf))
        # One invalid update must not poison this individual or its descendants.
        return jax.lax.cond(finite, lambda: (new_p, new_state), lambda: (p, state_i))

    batch_step = jax.jit(jax.vmap(step, in_axes=(0, 0, 0, None, None)))

    def score(p, a):
        # Scan images to avoid materializing population x validation x all paths.
        values = jax.lax.map(lambda pair: objective(p, a, pair[0], pair[1]),
                             (validation, val_targets))
        return jnp.mean(values)

    batch_score = jax.jit(jax.vmap(score))
    history = []
    best = None
    for generation in range(c.generations):
        for _ in range(c.gd_steps):
            index = int(rng.integers(len(train)))
            params, state = batch_step(params, state, arch, train[index], train_targets[index])
        # Score UPDATED parameters on the same validation images every time.
        scores = np.asarray(batch_score(params, arch))
        winner = best_candidate(scores)
        if best is None or scores[winner] < best.validation_loss:
            best = SearchResult(np.array(arch[winner], copy=True),
                                tuple(np.array(p[winner], copy=True) for p in params),
                                float(scores[winner]), history,
                                initial_arch, initial_params)
        history.append(best.validation_loss)
        if progress is not None:
            progress(generation + 1, best.validation_loss)
        if generation + 1 == c.generations:
            break  # Do not replace the evaluated final population with children.

        ranked = np.argsort(np.where(np.isfinite(scores), scores, np.inf))
        elite_indices = ranked[:min(max(1, c.pop_size // 4), int(np.isfinite(scores).sum()))]
        new_arch, new_params, new_states = [], [], []
        for i in elite_indices:
            new_arch.append(arch[i])
            new_params.append(tuple(p[i] for p in params))
            new_states.append(jax.tree_util.tree_map(lambda x: x[i], state))
        while len(new_arch) < c.pop_size:
            parent = int(rng.choice(elite_indices))
            child_arch = np.array(arch[parent], copy=True)
            if rng.random() < 0.2:
                block = int(rng.integers(c.num_blocks))
                child_arch[block] = (child_arch[block] + rng.integers(1, len(OPS))) % len(OPS)
            child = projection(tuple(p[parent] + jnp.asarray(rng.normal(0, scale, c.num_blocks),
                                     dtype=p.dtype) for p, scale in zip(params, (100., 100., 10.))))
            new_arch.append(jnp.asarray(child_arch))
            new_params.append(child)
            new_states.append(optimizer.init(child))
        arch = jnp.stack(new_arch)
        params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_params)
        state = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *new_states)
    return best


def evaluate(result, config, images):
    """Evaluate the saved winner on a batch that was not used for selection."""
    c = config
    images = _check_batch(images, c, "Evaluation images")
    def one(image):
        target = generate_target_image(c.shape, c.dx, c.spectrum, c.f1, c.f2, image, c.task)
        output = simulate_discrete_architecture(*result.parameters, result.architecture,
                    c.shape, c.dx, c.spectrum, image, c.pad_width, c.task)
        score = loss_fn(*result.parameters, result.architecture, c.shape, c.dx,
                       c.spectrum, image, target, c.pad_width, c.task)
        input_power = jnp.mean(image**2) if c.task == AMPLITUDE else 1.0
        throughput = jnp.sum(output) * c.dx**2 / jnp.maximum(input_power, 1e-20)
        return score, throughput, target, output
    scores, throughput, targets, outputs = jax.jit(jax.vmap(one))(images)
    return {"loss": float(jnp.mean(scores)), "throughput": float(jnp.mean(throughput)),
            "targets": np.asarray(targets), "outputs": np.asarray(outputs)}
