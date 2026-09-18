"""
Optical Architecture Search (OAS) - Example 5.1
Inverse Design of a 4f Coherent Imaging System

This script solves the inverse problem proposed in the document:
Given a task (imaging an amplitude object with magnification M) and a dataset (randomly 
generated amplitude masks), we define a generic optical architecture search space:
    Propagate(z1) -> PhaseMask(phi1) -> Propagate(z2) -> PhaseMask(phi2) -> Propagate(z3)

We jointly optimize the continuous parameters of the architecture:
- z1, z2, z3: the free-space propagation distances
- phi1, phi2: the spatially varying phase masks

By optimizing the MSE loss against the ideal target (from a simulated ideal 4f system),
the search may find a useful intensity transformation. Intensity-only agreement
does not identify unique phase masks or imply convergence to a classical 4f layout.
"""

import jax
import jax.numpy as jnp
import optax
import chromatix.functional as cx
import matplotlib.pyplot as plt

from chromatix import crop
from oas_core import generate_target_image, input_field, normalized_mse

def simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width):
    """
    Simulates the proposed continuous architecture search space.
    """
    phi1, phi2 = params['phi1'], params['phi2']
    z1, z2, z3 = (jnp.maximum(params[name], 1.0) for name in ('z1', 'z2', 'z3'))
    
    field = input_field(shape, dx, spectrum, a_mask, "5.1. Amplitude Imaging", pad_width)
    
    # Element 1: Free space propagation
    field = cx.transfer_propagate(field, z=z1, n=1.0, pad_width=0, mode="same")
    
    # Element 2: Learnable phase mask 1
    field = cx.phase_change(field, jnp.pad(phi1, pad_width))
    
    # Element 3: Free space propagation
    field = cx.transfer_propagate(field, z=z2, n=1.0, pad_width=0, mode="same")
    
    # Element 4: Learnable phase mask 2
    field = cx.phase_change(field, jnp.pad(phi2, pad_width))
    
    # Element 5: Free space propagation
    field = cx.transfer_propagate(field, z=z3, n=1.0, pad_width=0, mode="same")
    
    return crop(field, pad_width).intensity

def loss_fn(params, shape, dx, spectrum, a_mask, target_I, pad_width):
    I_out = simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width)
    
    # Normalized MSE loss
    mse_loss = normalized_mse(I_out, target_I)
    return mse_loss

def optimize():
    # Simulation settings
    shape = (64, 64)
    dx = 5.0  # pixel pitch
    spectrum = 0.532  # wavelength
    pad_width = 64
    
    # We define the ideal target using specific 4f configuration parameters
    target_f1 = 2000.0
    target_f2 = 4000.0  # M = 2x
    
    # Initialize parameters randomly or neutrally
    params = {
        'phi1': jnp.zeros(shape),
        'phi2': jnp.zeros(shape),
        'z1': jnp.array(1500.0), # Starting distances
        'z2': jnp.array(5000.0), 
        'z3': jnp.array(3500.0)  
    }
    
    # We use multiple learning rates: fast for distances, slower for phase
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
        # Step 1 & 2 of Dataset definition (Sec 3.2): 
        # Draw random instance object and compute target
        a_mask = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
        target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, a_mask)
        
        # Step 3: Compute loss and update
        loss, grads = jax.value_and_grad(loss_fn)(params, shape, dx, spectrum, a_mask, target_I, pad_width)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        for name in ('z1', 'z2', 'z3'):
            params[name] = jnp.clip(params[name], 1.0, 40000.0)
        return params, opt_state, loss

    epochs = 500
    print("Starting OAS Optimization Loop...")
    key = jax.random.PRNGKey(42)
    for i in range(epochs):
        key, subkey = jax.random.split(key)
        params, opt_state, loss = step(params, opt_state, subkey)
        
        if (i + 1) % 50 == 0:
            print(f"Epoch {i+1}/{epochs}, Loss: {float(loss):.4f} | Learned z1: {float(params['z1']):.1f}, z2: {float(params['z2']):.1f}, z3: {float(params['z3']):.1f}")

    print("Optimization complete!")
    
    # Evaluate final
    a_mask = jax.random.uniform(key, shape, minval=0.0, maxval=1.0)
    target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, a_mask)
    final_I = simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width)
    
    # Calculate what the ideal lens phase should look like for comparison
    field = cx.plane_wave(shape=shape, dx=dx, spectrum=spectrum, power=1.0)
    L = jnp.sqrt(spectrum * target_f1 / 1.0)
    import chromatix.utils
    ideal_phi1 = (-jnp.pi * chromatix.utils.l2_sq_norm(field.grid) / L**2).squeeze()
    
    # Plot results
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
    
    plt.subplot(1, 4, 4)
    plt.title("Ideal Phase Mask 1")
    plt.imshow(ideal_phi1, cmap='viridis')
    plt.colorbar()
    
    plt.savefig("inverse_design_output.png")
    print("Saved plot to inverse_design_output.png")

if __name__ == "__main__":
    optimize()
