"""Scalar coherent forward model and exact expected error for an affine readout."""
import chromatix.functional as cx
from chromatix import crop, pad
import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
import numpy as np
from oas_core import OPS, apply_operation
from .data import roi

ELEMENTS = OPS + ("PhaseDisk", "FreeformPhase")
FULL_LIBRARY = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
RESTRICTED_LIBRARY = (1, 2, 3, 12)
HARD_WIDTHS = (3, 4, 7, 12)


def bounded(raw, low, high):
    return low + (high - low) * jax.nn.sigmoid(raw)


def coordinates(config):
    extent = config.size + 2*config.padding
    axis = (jnp.arange(extent)-extent//2)*config.pitch_um
    return jnp.meshgrid(axis, axis, indexing="ij")


def tukey_profile(width, config):
    """Flat core of diameter width; cosine taper to the physical mount edge."""
    y, x = coordinates(config)
    radius = jnp.sqrt(x*x+y*y)
    outer = config.clear_aperture_um/2
    inner = jnp.minimum(width/2, outer)
    fraction = jnp.clip((radius-inner)/jnp.maximum(outer-inner, 1e-6), 0., 1.)
    return jnp.where(radius <= inner, 1., .5*(1+jnp.cos(jnp.pi*fraction)))


def grating_phase(operation, width, config):
    """Physical phase profiles anchored at the optical axis, never an FFT edge."""
    _, x = coordinates(config)
    period = jnp.maximum(width/2, config.min_grating_period_um)
    return jnp.where(operation == 10, 2*jnp.pi*(x % period)/period,
                     jnp.pi*jnp.sin(2*jnp.pi*x/period))


def physical(parameters, config):
    c = config
    spare = c.length_um - (c.stations + 1) * c.min_gap_um
    return {"gaps_um": c.min_gap_um + spare * jax.nn.softmax(parameters["gaps"]),
            "focal_um": bounded(parameters["focal"], c.focal_min_um, c.focal_max_um),
            "width_um": bounded(parameters["width"], c.width_min_um, c.width_max_um),
            "phase_rad": jnp.pi * jnp.tanh(parameters["phase"])}


def freeform_phase(mask, config):
    c = config
    y, x = coordinates(c)
    coords = jnp.stack([(y/c.clear_aperture_um+.5)*(c.mask_grid-1),
                        (x/c.clear_aperture_um+.5)*(c.mask_grid-1)])
    return jnp.pi*jnp.tanh(map_coordinates(mask, coords, order=1, mode="nearest"))


def intensity(parameters, architecture, phase, config, defocus_um=0.0):
    """Intensity relative to incident intensity, with NO per-image normalization.

    Propagation precedes every station and follows the last. Empty stations
    still occupy axial coordinates. Padding is zero field, not extra photons.
    Detector samples the cropped field at the original pitch (point samples).
    """
    c = config
    p = physical(parameters, c)
    field = cx.plane_wave(shape=(c.size, c.size), dx=c.pitch_um,
                          spectrum=c.wavelength_um, power=1.)
    field = pad(cx.phase_change(field, phase), c.padding)
    y, x = coordinates(c)
    r2 = x*x + y*y
    clear = r2 <= (c.clear_aperture_um/2)**2

    def station(fld, values):
        op, gap, focal, width, height, mask = values
        fld = cx.transfer_propagate(fld, z=gap, n=1., pad_width=0, mode="same")
        # All installed elements have a finite common mount aperture and loss.
        fld = jax.lax.cond(op == 2, lambda f: f,
            lambda f: cx.amplitude_change(f, clear * jnp.sqrt(c.element_transmission)), fld)

        def conventional(f):
            # Freeze the legacy grating/axicon parameterization in physical
            # units, so refining the numerical pitch does not change a design.
            return jax.lax.cond(op == 8,
                lambda g: cx.amplitude_change(g, tukey_profile(width, c)),
                lambda g: jax.lax.cond((op == 10) | (op == 11),
                    lambda h: cx.phase_change(h, grating_phase(op, width, c)),
                    lambda h: apply_operation(h, op, 0., focal, width,
                        c.min_grating_period_um/8, c.wavelength_um), g), f)

        def plate(f):
            return cx.phase_change(f, height * (r2 <= (width/2)**2))

        def freeform(f):
            # The physical mask extent is fixed by its aperture, independent
            # of FFT padding and numerical pitch (essential for convergence).
            return cx.phase_change(f, freeform_phase(mask, c))

        fld = jax.lax.cond(op == 13, freeform,
            lambda f: jax.lax.cond(op == 12, plate, conventional, f), fld)
        return fld, None

    field, _ = jax.lax.scan(station, field, (architecture, p["gaps_um"][:-1],
        p["focal_um"], p["width_um"], p["phase_rad"], parameters["masks"]))
    field = cx.transfer_propagate(field, z=p["gaps_um"][-1]+defocus_um,
                                  n=1., pad_width=0, mode="same")
    return crop(field, c.padding).intensity * (c.size**2 * c.pitch_um**2)


def noise_variance(signal, photons, config, noise_aware=True):
    """Variance after electron measurements are divided by incident D*QE."""
    if not noise_aware:
        return jnp.zeros_like(signal)
    scale = photons * config.quantum_efficiency
    return jnp.maximum(signal, 0.) / scale + (config.read_noise_e / scale)**2


def orient(signal, inverted):
    # Invert about the optical axis (index N/2), not between the central pixels.
    flipped = jnp.roll(jnp.flip(signal, axis=(-2, -1)), (1, 1), axis=(-2, -1))
    return jnp.where(inverted, flipped, signal)


def calibrate(signals, targets, config, noise_aware=True):
    """Training-only analytic ridge fit of shared gain/offset and orientation.

    signals may include a leading defocus-ensemble dimension. Orientation is
    selected on training risk, with identical options for every baseline.
    """
    weight = jnp.asarray(roi(config.size), dtype=jnp.float32)
    weight = jnp.broadcast_to(weight, signals.shape)
    targets = jnp.broadcast_to(targets, signals.shape)
    mean = lambda x: jnp.sum(weight*x) / jnp.sum(weight)

    def one(inverted):
        s = orient(signals, inverted)
        mu_s, mu_t = mean(s), mean(targets)
        variance = noise_variance(s, config.photons, config, noise_aware)
        a = mean((s-mu_s)*(targets-mu_t)) / (mean((s-mu_s)**2 + variance) + config.calibration_ridge)
        b = mu_t - a*mu_s
        risk = mean((a*s+b-targets)**2 + a*a*variance)
        return risk, a, b

    values, gains, offsets = jax.vmap(one)(jnp.array([False, True]))
    best = jnp.argmin(values)
    return {"gain": gains[best], "offset": offsets[best], "inverted": best == 1}


def expected_mse(signals, targets, calibration, config, photons=None, noise_aware=True):
    signal = orient(signals, calibration["inverted"])
    prediction = calibration["gain"] * signal + calibration["offset"]
    variance = noise_variance(signal, config.photons if photons is None else photons,
                              config, noise_aware)
    error = (prediction - targets)**2 + calibration["gain"]**2 * variance
    weight = jnp.asarray(roi(config.size))
    return jnp.sum(error * weight, axis=(-2, -1)) / jnp.sum(weight)


def sample_camera(signal, photons, config, rng):
    """Exact Poisson shot noise plus independent Gaussian read noise, in electrons.

    No clipping, saturation, dark current, EM gain or ADC quantization is modeled.
    Negative read-noise samples are retained.
    """
    mean = np.maximum(np.asarray(signal), 0) * photons * config.quantum_efficiency
    return rng.poisson(mean) + rng.normal(0, config.read_noise_e, mean.shape)


def prescription(parameters, architecture, config):
    p = {k: np.asarray(v) for k, v in physical(parameters, config).items()}
    result = []
    position = 0.
    for i, op in enumerate(architecture):
        position += float(p["gaps_um"][i])
        item = {"station": i, "operation": ELEMENTS[int(op)], "position_um": position,
                "preceding_gap_um": float(p["gaps_um"][i])}
        if op != 2:
            item.update(clear_aperture_um=config.clear_aperture_um,
                        intensity_transmission=config.element_transmission)
        if op == 1:
            item["focal_length_um"] = float(p["focal_um"][i])
        elif op in (3, 4, 5, 6, 7, 8, 12):
            item["width_um"] = float(p["width_um"][i])
            if op == 7:
                item["height_um"] = item["width_um"] / 2
            if op == 12:
                item["phase_shift_rad"] = float(p["phase_rad"][i])
        elif op == 9:
            w = float(p["width_um"][i])
            item["slope_angle_rad"] = w/(w+100) * min(.1, 4*config.wavelength_um/config.min_grating_period_um)
            item["refractive_index"] = 1.5
        elif op in (10, 11):
            item["period_um"] = float(max(p["width_um"][i]/2, config.min_grating_period_um))
            item["thickness_um"] = config.wavelength_um * (2 if op == 10 else 1)
            item["refractive_index"] = 1.5
        elif op == 13:
            item["mask"] = f"parameters.npz:masks[{i}] (bilinear on clear aperture then pi*tanh)"
        result.append(item)
    return {"stations": result, "sensor_gap_um": float(p["gaps_um"][-1]),
            "length_um": float(np.sum(p["gaps_um"])),
            "component_count": int(np.count_nonzero(np.asarray(architecture) != 2))}


def sampling_warnings(parameters, architecture, config):
    p = {k: np.asarray(v) for k, v in physical(parameters, config).items()}
    messages = []
    threshold = (config.size + 2*config.padding) * config.pitch_um**2 / config.wavelength_um
    if np.any(p["gaps_um"] > threshold):
        messages.append("A Fresnel transfer kernel exceeds the adjacent-phase sampling bound.")
    lens_limit = config.clear_aperture_um * config.pitch_um / config.wavelength_um
    if np.any(p["focal_um"][np.asarray(architecture) == 1] < lens_limit):
        messages.append("A lens phase is undersampled within its clear aperture.")
    messages.append("Require fixed-design padding and pixel-pitch convergence before scientific claims.")
    return messages
