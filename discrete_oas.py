"""
Discrete Optical Architecture Search (OAS)
Experimental straight-through Gumbel search (not the default memetic search).

Instead of a fixed sequence of propagations and phase masks, 
this algorithm selects the optical elements from a predefined library.
"""

import jax
import jax.numpy as jnp
import optax
import chromatix.functional as cx
import numpy as np
from chromatix import crop
from oas_data import load_cifar_batch
from oas_core import (
    generate_target_image as sensor_target, input_field, normalized_mse,
    project_parameters,
)

# Our library of discrete optical elements
OPS = ["Propagate", "ThinLens", "Identity"]

def generate_target_image(shape, dx, spectrum, f1, f2, a_mask, pad_width):
    return sensor_target(shape, dx, spectrum, f1, f2, a_mask)

def gumbel_softmax(logits, key, temperature=1.0, hard=True):
    gumbels = -jnp.log(-jnp.log(jax.random.uniform(key, logits.shape) + 1e-20) + 1e-20)
    y_soft = jax.nn.softmax((logits + gumbels) / temperature)
    
    if hard:
        # Straight-through estimator
        index = jnp.argmax(y_soft, axis=-1)
        y_hard = jax.nn.one_hot(index, logits.shape[-1])
        y = jax.lax.stop_gradient(y_hard - y_soft) + y_soft
        return y
    return y_soft

def simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width, tau, key, deterministic=False):
    field = input_field(shape, dx, spectrum, a_mask, "5.1. Amplitude Imaging", pad_width)
    logits = params['arch_logits']
    z, f, _ = project_parameters((params['z'], params['f'], jnp.ones_like(params['z'])),
                                 shape, dx, spectrum, pad_width)

    keys = jax.random.split(key, logits.shape[0])
    
    for i in range(logits.shape[0]):
        # Temperature does not remove Gumbel randomness. Exported architectures
        # must be evaluated with exactly the reported argmax operation.
        w = (jax.nn.one_hot(jnp.argmax(logits[i]), len(OPS)) if deterministic
             else gumbel_softmax(logits[i], keys[i], temperature=tau, hard=True))
        
        u_prop = cx.transfer_propagate(field, z=z[i], n=1.0, pad_width=0, mode="same").u
        u_lens = cx.thin_lens(field, f=f[i], n=1.0).u
        u_id = field.u
        
        u_new = w[0] * u_prop + w[1] * u_lens + w[2] * u_id
        field = field.replace(u=u_new)
        
    return crop(field, pad_width).intensity

def loss_fn(params, shape, dx, spectrum, a_mask, target_I, pad_width, tau, key):
    I_out = simulate_search_architecture(params, shape, dx, spectrum, a_mask, pad_width, tau, key)
    mse_loss = normalized_mse(I_out, target_I)
    return mse_loss

def optimize():
    shape = (64, 64)
    dx = 5.0  
    spectrum = 0.532  
    pad_width = 64
    
    target_f1 = 4000.0
    target_f2 = 4000.0
    
    num_blocks = 7 # Give it slightly more capacity than the minimum 5 blocks needed
    
    key = jax.random.PRNGKey(42)
    key, key_z, key_f, key_logits = jax.random.split(key, 4)
    f_min = float(project_parameters((jnp.array(1.), jnp.array(1.), jnp.array(1.)),
                                     shape, dx, spectrum, pad_width)[1])
    
    # Initialize parameters
    params = {
        # Architecture routing logits (start neutral with slight noise for symmetry breaking)
        'arch_logits': jax.random.normal(key_logits, (num_blocks, 3)) * 0.01,
        # Resolve each lens's own phase ramp on the working grid.
        'z': jax.random.uniform(key_z, (num_blocks,), minval=1000.0, maxval=8000.0),
        'f': jax.random.uniform(key_f, (num_blocks,), minval=f_min, maxval=2*f_min),
    }
    
    params['z'], params['f'], _ = project_parameters(
        (params['z'], params['f'], jnp.ones(num_blocks)), shape, dx, spectrum, pad_width)
    optimizer = optax.multi_transform(
        {'arch': optax.adam(0.5), 'phys': optax.adam(10.0)},
        param_labels={
            'arch_logits': 'arch',
            'z': 'phys', 'f': 'phys'
        }
    )
    opt_state = optimizer.init(params)
    
    @jax.jit
    def step(params, opt_state, a_mask, tau, key):
        target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, a_mask, pad_width)
        
        loss, grads = jax.value_and_grad(loss_fn)(params, shape, dx, spectrum, a_mask, target_I, pad_width, tau, key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params['z'], params['f'], _ = project_parameters(
            (params['z'], params['f'], jnp.ones(num_blocks)), shape, dx, spectrum, pad_width)
        return params, opt_state, loss

    epochs = 1000
    print("Loading CIFAR-10 data...")
    
    # We will use a batch of images to train the architecture
    cifar_batch = load_cifar_batch(batch_size=10, img_size=shape[0])
    
    print("Starting Discrete Architecture Search on CIFAR-10...")
    
    for i in range(epochs):
        key, sample_key, route_key = jax.random.split(key, 3)
        # Randomly select one image from our loaded batch for stochastic gradient descent
        img_idx = jax.random.randint(sample_key, (), 0, cifar_batch.shape[0])
        a_mask = cifar_batch[img_idx]
        
        # Temperature annealing
        tau = max(0.1, 1.0 * (0.98 ** i))
        
        params, opt_state, loss = step(params, opt_state, a_mask, tau, route_key)
        
        if (i + 1) % 100 == 0:
            arch_indices = np.argmax(params['arch_logits'], axis=-1)
            proposed_arch = [OPS[idx] for idx in arch_indices]
            
            print(f"\n--- Epoch {i+1}/{epochs} | Loss: {float(loss):.4f} | Temp: {tau:.3f} ---")
            print("Current Architecture Proposal:")
            for b, (op, z_val, f_val) in enumerate(zip(proposed_arch, params['z'], params['f'])):
                param_str = f"z={float(z_val):.1f}" if op == "Propagate" else (f"f={float(f_val):.1f}" if op == "ThinLens" else "None")
                print(f"  Block {b}: {op} ({param_str})")

    print("\n--- Final Evaluation ---")
    arch_indices = np.argmax(params['arch_logits'], axis=-1)
    proposed_arch = [OPS[idx] for idx in arch_indices]
    print("Final Chosen Architecture:")
    for b, (op, z_val, f_val) in enumerate(zip(proposed_arch, params['z'], params['f'])):
        param_str = f"z={float(z_val):.1f}" if op == "Propagate" else (f"f={float(f_val):.1f}" if op == "ThinLens" else "None")
        print(f"  Block {b}: {op} ({param_str})")
        
    # Test pass on a completely unseen image
    test_key, _ = jax.random.split(key)
    test_batch = load_cifar_batch(batch_size=1, img_size=shape[0], split="test", seed=42)
    test_mask = test_batch[0]
    
    target_I = generate_target_image(shape, dx, spectrum, target_f1, target_f2, test_mask, pad_width)
    
    # Evaluate the reported architecture without Gumbel noise.
    final_I = simulate_search_architecture(params, shape, dx, spectrum, test_mask, pad_width, 0.01, test_key, deterministic=True)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Object Amplitude Mask")
    plt.imshow(test_mask, cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.title("Target Intensity (Ideal 4f)")
    plt.imshow(target_I.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.title("Learned Architecture Output")
    plt.imshow(final_I.squeeze(), cmap='gray')
    plt.colorbar()
    
    plt.savefig("discrete_oas_output.png")
    print("Saved evaluation plot to discrete_oas_output.png")

if __name__ == "__main__":
    optimize()
