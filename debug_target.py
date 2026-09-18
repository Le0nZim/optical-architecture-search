"""Compare sampled thin-lens propagation with a target on the same sensor grid."""

import jax.numpy as jnp
import matplotlib.pyplot as plt

from oas_core import generate_target_image, sampling_warnings, simulate_discrete_architecture
from oas_data import load_cifar_batch


def main():
    shape, dx, spectrum = (64, 64), 5., .532
    f1, f2, padding = 500., 500., 64
    obj = load_cifar_batch(1, shape[0], split="test", seed=42)[0]
    architecture = jnp.array([0, 1, 0, 1, 0])
    parameters = (jnp.array([f1, 1., f1+f2, 1., f2]),
                  jnp.array([1., f1, 1., f2, 1.]), jnp.ones(5)*100.)
    # Deliberately undersampled legacy settings: print that fact explicitly.
    for message in sampling_warnings(architecture, parameters, shape, dx, spectrum, padding):
        print(message)
    simulated = simulate_discrete_architecture(*parameters, architecture, shape,
                                                dx, spectrum, obj, padding)
    target = generate_target_image(shape, dx, spectrum, f1, f2, obj)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, image, title in zip(axes, (obj, simulated, target),
            ("Input mask", "Sampled thin-lens steps", "Ideal 4f on the same sensor grid")):
        ax.imshow(image, cmap="gray")
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig("target_debug.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
