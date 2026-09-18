import jax
import jax.numpy as jnp
import optax
import chromatix.functional as cx
import numpy as np
import matplotlib.pyplot as plt
import os

from chromatix import crop
from oas_core import generate_target_image, input_field, normalized_mse

def simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width):
    """
    Simulates the proposed architecture:
    Propagate z1 -> PhaseMask1 -> Propagate z2 -> PhaseMask2 -> Propagate z3
    """
    phi1 = params['phi1']
    phi2 = params['phi2']
    z1 = jnp.maximum(params['z1'], 1.0)
    z2 = jnp.maximum(params['z2'], 1.0)
    z3 = jnp.maximum(params['z3'], 1.0)
    
    field = input_field(shape, dx, spectrum, a_mask, "5.1. Amplitude Imaging", pad_width)
    
    # Propagate z1
    field = cx.transfer_propagate(field, z=z1, n=1.0, pad_width=0, mode="same")
    
    # Phase Mask 1
    field = cx.phase_change(field, jnp.pad(phi1, pad_width))
    
    # Propagate z2
    field = cx.transfer_propagate(field, z=z2, n=1.0, pad_width=0, mode="same")
    
    # Phase Mask 2
    field = cx.phase_change(field, jnp.pad(phi2, pad_width))
    
    # Propagate z3
    field = cx.transfer_propagate(field, z=z3, n=1.0, pad_width=0, mode="same")
    
    return crop(field, pad_width).intensity

def loss_fn(params, shape, dx, spectrum, a_mask, target_I, pad_width):
    I_out = simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width)
    
    # MSE loss scaled up
    mse_loss = normalized_mse(I_out, target_I)
              
    return mse_loss

def optimize():
    shape = (64, 64)
    dx = 5.0  # pixel pitch
    spectrum = 0.532  # wavelength
    
    # 4f configuration parameters
    f1 = 2000.0
    f2 = 4000.0
    
    z1 = f1
    z2 = f1 + f2
    z3 = f2
    
    pad_width = 64
    
    # Initialize parameters
    params = {
        'phi1': jnp.zeros(shape),
        'phi2': jnp.zeros(shape),
        'z1': jnp.array(1500.0), # Start somewhat close to 2000
        'z2': jnp.array(5000.0), # Start somewhat close to 6000
        'z3': jnp.array(3500.0)  # Start somewhat close to 4000
    }
    
    # Different learning rates for phase masks and distances
    optimizer = optax.multi_transform(
        {'phase': optax.adam(0.1), 'dist': optax.adam(50.0)},
        param_labels={
            'phi1': 'phase', 'phi2': 'phase',
            'z1': 'dist', 'z2': 'dist', 'z3': 'dist'
        }
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state, key):
        a_mask = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
        target_I = generate_target_image(shape, dx, spectrum, f1, f2, a_mask)
        loss, grads = jax.value_and_grad(loss_fn)(params, shape, dx, spectrum, a_mask, target_I, pad_width)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        for name in ('z1', 'z2', 'z3'):
            params[name] = jnp.clip(params[name], 1.0, 40000.0)
        return params, opt_state, loss

    epochs = 500
    print("Starting optimization...")
    key = jax.random.PRNGKey(42)
    for i in range(epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey)
        
        if (i + 1) % 50 == 0:
            print(f"Epoch {i+1}/{epochs}, Loss: {float(loss):.6f}, z1: {float(params['z1']):.1f}, z2: {float(params['z2']):.1f}, z3: {float(params['z3']):.1f}")

    print("Optimization complete!")
    
    # Evaluate final
    a_mask = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
    target_I = generate_target_image(shape, dx, spectrum, f1, f2, a_mask)
    final_I = simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width)
    
    print(f"Target max intensity: {float(jnp.max(target_I)):.6e}")
    print(f"Final max intensity: {float(jnp.max(final_I)):.6e}")
    
    plt.figure(figsize=(20, 5))
    plt.subplot(1, 4, 1)
    plt.title("Target Intensity (4f)")
    plt.imshow(target_I.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 4, 2)
    plt.title("Learned Architecture Output")
    plt.imshow(final_I.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 4, 3)
    plt.title("Learned Phase Mask 1")
    plt.imshow(params['phi1'].squeeze(), cmap='viridis')
    plt.colorbar()
    
    # Ideal lens 1 phase
    field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
    L = jnp.sqrt(spectrum * f1 / 1.0)
    import chromatix.utils
    ideal_phi1 = -jnp.pi * chromatix.utils.l2_sq_norm(field.grid) / L**2
    ideal_phi1 = ideal_phi1.squeeze()
    
    # We need to compute phase difference modulo 2pi
    phase_diff = (params['phi1'].squeeze() - ideal_phi1) % (2 * jnp.pi)
    phase_diff = jnp.minimum(phase_diff, 2 * jnp.pi - phase_diff)
    print(f"Mean phase difference (mod 2pi) for Mask 1: {float(jnp.mean(phase_diff)):.4f} rad")
    
    plt.subplot(1, 4, 4)
    plt.title("Ideal Phase Mask 1")
    plt.imshow(ideal_phi1, cmap='viridis')
    plt.colorbar()
    
    plt.savefig("inverse_design_output.png")
    print("Saved plot to inverse_design_output.png")

if __name__ == "__main__":
    optimize()
