import importlib

import chromatix.functional as cx
from chromatix.utils.initializers import sawtooth_phase, sinusoid_phase
from chromatix.utils.utils import create_grid
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oas_core import (
    AMPLITUDE, PHASE, OPS, generate_target_image, normalized_mse, standardized,
    simulate_discrete_architecture, loss_fn, project_parameters,
    describe_architecture, sampling_warnings,
)


def test_relative_loss_uses_physical_intensity_scale():
    target = jnp.full((8, 8), 1e-6)
    assert float(normalized_mse(jnp.zeros_like(target), target)) == pytest.approx(1)
    assert float(normalized_mse(target, target)) == 0
    assert float(normalized_mse(jnp.zeros_like(target), jnp.zeros_like(target))) == 0
    with pytest.raises(ValueError):
        normalized_mse(target, jnp.ones((1, 8, 8)))


def test_flat_contrast_has_finite_gradient_and_ignores_global_offset():
    image = jnp.ones((8, 8))
    target = jnp.linspace(0, 1, 64).reshape(8, 8)
    loss = lambda x: jnp.mean((standardized(x) - standardized(target))**2)
    value, grad = jax.value_and_grad(loss)(image)
    assert np.isfinite(value) and np.isfinite(grad).all()
    np.testing.assert_allclose(standardized(target), standardized(target + 2), atol=2e-6)


@pytest.mark.parametrize('ratio', [0.5, 1.0, 2.0])
def test_ideal_target_has_correct_magnification_and_radiometry(ratio):
    shape = (64, 64)
    yy, xx = jnp.meshgrid(jnp.arange(64) - 32, jnp.arange(64) - 32, indexing='ij')
    # Off-axis asymmetric smooth object gives an independent analytic oracle.
    mask = jnp.exp(-((xx - 6)**2 + (yy + 3)**2) / (2 * 5**2))
    target = generate_target_image(shape, 1., .532, 4000., 4000. * ratio, mask)
    expected = jnp.exp(-((-xx / ratio - 6)**2 + (-yy / ratio + 3)**2) / 5**2)
    expected = expected / (64**2 * ratio**2)
    relative_error = np.linalg.norm(target - expected) / np.linalg.norm(expected)
    assert relative_error < 0.013
    row, col = np.unravel_index(np.argmax(target), shape)
    assert abs(row - (32 + 3 * ratio)) <= 1
    assert abs(col - (32 - 6 * ratio)) <= 1


def test_free_space_composition_does_not_insert_intermediate_field_stop():
    shape = (16, 16)
    obj = jnp.zeros(shape).at[2:6, 2:6].set(1.)
    def simulate(z):
        z = jnp.asarray(z)
        return simulate_discrete_architecture(z, jnp.ones_like(z)*4000,
            jnp.ones_like(z)*30, jnp.zeros(len(z), dtype=jnp.int32),
            shape, 5., .532, obj, 16, AMPLITUDE)
    a, b = simulate([3000., 5000.]), simulate([8000.])
    np.testing.assert_allclose(a, b, rtol=3e-5, atol=2e-10)


def test_target_does_not_wrap_out_of_view_object_to_opposite_border():
    mask = jnp.zeros((8, 8)).at[0, 3].set(1.)
    # y=-4 maps to y=+4, outside a sensor whose last sample is y=+3.
    target = generate_target_image((8, 8), 1., .532, 4000., 4000., mask)
    np.testing.assert_array_equal(target, np.zeros((8, 8)))


@pytest.mark.parametrize('task', [AMPLITUDE, PHASE])
def test_all_operations_have_finite_batched_gradients(task):
    shape = (8, 8)
    image = jnp.asarray(np.random.default_rng(7).random(shape), dtype=jnp.float32)
    target = generate_target_image(shape, 5., .532, 4000., 4000., image, task)
    parameters = tuple(jnp.full((len(OPS), 1), x) for x in (1000., 5000., 80.))
    architectures = jnp.arange(len(OPS), dtype=jnp.int32)[:, None]
    def objective(params, arch):
        return loss_fn(*params, arch, shape, 5., .532, image, target, 4, task)
    values, grads = jax.jit(jax.vmap(jax.value_and_grad(objective)))(parameters, architectures)
    assert np.isfinite(values).all()
    assert all(np.isfinite(g).all() for g in grads)


def test_tukey_at_full_width_is_finite_even_under_vmap():
    field = cx.plane_wave(shape=(16, 16), dx=5., spectrum=.532, power=1.)
    def loss(w):
        return jnp.sum(cx.tukey_pupil(field, w).intensity)
    values, grads = jax.jit(jax.vmap(jax.value_and_grad(loss)))(jnp.array([79., 80., 100.]))
    assert np.isfinite(values).all() and np.isfinite(grads).all()
    np.testing.assert_allclose(cx.tukey_pupil(field, 80.).u, field.u)


def test_grating_phase_uses_physical_period_not_sample_maximum():
    shape = (8, 8)
    spacing, period, wavelength, thickness = 1., 3.7, .532, .8
    grid = create_grid(shape, spacing)
    coordinate = grid[1] - grid[1].min()
    scale = 2 * jnp.pi * .5 * thickness / wavelength
    actual = sawtooth_phase(shape, spacing, wavelength, 1.5, period, thickness)
    np.testing.assert_allclose(actual, scale * (coordinate % period) / period, rtol=1e-6)
    actual = sinusoid_phase(shape, spacing, wavelength, 1.5, period, thickness)
    np.testing.assert_allclose(actual, scale * jnp.sin(2*jnp.pi*coordinate/period), rtol=1e-6)
    # Previously every sampled sawtooth height was zero, leading to 0/0.
    assert np.isfinite(sawtooth_phase(shape, 1., wavelength, 1.5, 1., thickness)).all()


def test_hard_pupil_widths_are_explicitly_derivative_free():
    field = cx.plane_wave(shape=(16, 16), dx=5., spectrum=.532, power=1.)
    for pupil in (cx.circular_pupil, cx.square_pupil):
        assert float(jax.grad(lambda w: jnp.sum(pupil(field, w).intensity))(31.)) == 0
        assert float(jnp.sum(pupil(field, 60.).intensity)) > float(jnp.sum(pupil(field, 20.).intensity))


def test_projected_values_and_export_are_the_simulated_values():
    shape, dx, wavelength, padding = (16, 16), 5., .532, 16
    arch = jnp.arange(len(OPS), dtype=jnp.int32)
    params = project_parameters(tuple(jnp.full(len(OPS), v) for v in (-5., -10., -15.)),
                                shape, dx, wavelength, padding)
    assert all(np.all(p > 0) for p in params)
    record = describe_architecture(arch, params, dx, wavelength)
    assert record[0]['z_um'] == float(params[0][0])
    assert record[1]['f_um'] == float(params[1][1])
    assert 'slope_angle_rad' in record[9]
    assert record[10]['period_um'] >= 8 * dx
    assert not sampling_warnings(arch, params, shape, dx, wavelength, padding)


def test_gumbel_export_is_deterministic_and_matches_reported_architecture():
    from discrete_oas import simulate_search_architecture
    shape = (8, 8)
    obj = jnp.asarray(np.random.default_rng(9).random(shape), dtype=jnp.float32)
    # Small logit margins make stochastic routing very different from argmax.
    params = {'arch_logits': jnp.array([[0.01, 0., 0.], [0., 0., 0.01]]),
              'z': jnp.array([2000., 3000.]), 'f': jnp.array([4000., 4000.])}
    outputs = [simulate_search_architecture(params, shape, 5., .532, obj, 4, .01,
                   jax.random.PRNGKey(seed), deterministic=True) for seed in (1, 2)]
    expected = simulate_discrete_architecture(params['z'], params['f'], jnp.ones(2),
        jnp.array([0, 2]), shape, 5., .532, obj, 4, AMPLITUDE)
    np.testing.assert_allclose(outputs[0], expected, rtol=1e-5, atol=1e-10)
    np.testing.assert_array_equal(outputs[0], outputs[1])


@pytest.mark.parametrize('module', ['inverse_design', 'oas_inverse_design'])
def test_continuous_examples_share_sensor_target_and_compose_free_space(module):
    example = importlib.import_module(module)
    shape = (8, 8)
    obj = jnp.asarray(np.random.default_rng(19).random(shape), dtype=jnp.float32)
    params = {'phi1': jnp.zeros(shape), 'phi2': jnp.zeros(shape),
              'z1': jnp.array(500.), 'z2': jnp.array(1000.), 'z3': jnp.array(1500.)}
    expected = simulate_discrete_architecture(jnp.array([3000.]), jnp.array([4000.]),
        jnp.array([20.]), jnp.array([0]), shape, 5., .532, obj, 4, AMPLITUDE)
    output = example.simulate_search_architecture(params, shape, 5., .532, obj, 4)
    np.testing.assert_allclose(output, expected, rtol=1e-5, atol=1e-10)
    target = example.generate_target_image(shape, 5., .532, 2000., 4000., obj)
    value, grads = jax.value_and_grad(example.loss_fn)(params, shape, 5., .532, obj, target, 4)
    assert np.isfinite(value) and all(np.isfinite(x).all() for x in grads.values())
