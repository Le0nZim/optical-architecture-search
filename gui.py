"""Streamlit front end for the tested, headless memetic search."""

import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import streamlit as st

from oas_core import (
    AMPLITUDE, PHASE, SearchConfig, describe_architecture, evaluate, run_search,
    sampling_warnings,
)
from oas_data import load_cifar_batch


@st.cache_data
def cached_cifar(batch_size, img_size, split, seed):
    return load_cifar_batch(batch_size, img_size, split=split, seed=seed)


def draw_schematic(ax, architecture, title="Optical System"):
    current_z = 0.0
    ax.axhline(0, color="black", linestyle="-.", linewidth=1)
    ax.axvline(0, color="green", linewidth=3)
    ax.text(0, 1.2, "Object", rotation=90, ha="center", color="green")
    for element in architecture:
        op = element["operation"]
        if op == "Propagate":
            distance = element["z_um"]
            ax.annotate("", xy=(current_z + distance, 0.5), xytext=(current_z, 0.5),
                        arrowprops={"arrowstyle": "<|-|>", "color": "gray"})
            ax.text(current_z + distance / 2, 0.6, f"{distance:.0f} µm", ha="center")
            current_z += distance
        elif op != "Identity":
            color = "blue" if op == "ThinLens" else ("orange" if "Pupil" in op else "purple")
            ax.axvline(current_z, ymin=0.25, ymax=0.75, color=color, linewidth=2)
            ax.text(current_z, -1.1, op, rotation=90, va="top", ha="center", color=color)
    ax.axvline(current_z, color="red", linewidth=3)
    ax.text(current_z, 1.2, "Sensor", rotation=90, ha="center", color="red")
    extent = max(current_z, 100)
    ax.set_xlim(-0.1 * extent, 1.1 * extent)
    ax.set_ylim(-3, 2)
    ax.axis("off")
    ax.set_title(title)


def main():
    st.set_page_config(page_title="Memetic OAS GUI", layout="wide")
    st.title("Memetic Optical Architecture Search (OAS)")
    st.write("Evolutionary element selection with continuous parameter optimization.")
    task = st.selectbox("Select Imaging Task", [AMPLITUDE, PHASE])
    if task == PHASE:
        st.info("Weak relative phase contrast (0.5 rad range). Absolute/global phase is not observable. "
                "The contrast objective does not model photon noise; sensor throughput is reported separately.")
    with st.sidebar:
        st.header("Search Settings")
        f1 = st.number_input("Target f1 (µm)", min_value=1.0, value=4000.0, step=100.0)
        f2 = st.number_input("Target f2 (µm)", min_value=1.0, value=4000.0, step=100.0)
        blocks = st.number_input("Number of elements", min_value=1, value=7, step=1)
        population = st.number_input("Population size", min_value=1, value=15, step=1)
        steps = st.number_input("Gradient steps per generation", min_value=1, value=5, step=1)
        generations = st.slider("Generations", 1, 200, 60)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        padding = st.number_input("Working-grid padding (pixels per side)", min_value=0, value=64, step=16)
        run = st.button("Run Architecture Search", type="primary")
    if run:
        config = SearchConfig(f1=f1, f2=f2, task=task, num_blocks=blocks,
                              pop_size=population, gd_steps=steps, generations=generations,
                              seed=seed, pad_width=padding)
        status = st.empty()
        bar = st.progress(0.)
        try:
            status.write("Loading disjoint CIFAR-10 training and validation samples…")
            train = cached_cifar(16, config.shape[0], "train", seed)
            validation = cached_cifar(8, config.shape[0], "validation", seed)
            def progress(generation, loss):
                bar.progress(generation / generations)
                status.write(f"Generation {generation}/{generations} | Best validation loss: {loss:.5g}")
            result = run_search(config, train, validation, progress)
            status.write("Evaluating the saved winner on the official CIFAR-10 test split…")
            test = cached_cifar(16, config.shape[0], "test", seed)
            performance = evaluate(result, config, test)
            st.session_state["oas_result"] = (config, result, test, performance)
            status.success("Search complete")
        except (ValueError, FloatingPointError, ImportError, RuntimeError, OSError) as exc:
            st.error(str(exc))
    if "oas_result" not in st.session_state:
        st.info("Choose your settings and run the search.")
        return
    config, result, test, performance = st.session_state["oas_result"]
    st.caption(f"Saved run: {config.task}; seed {config.seed}. Metrics below belong to this run.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Best validation loss", f"{result.validation_loss:.5g}")
    col2.metric("Held-out test loss (16 images)", f"{performance['loss']:.5g}")
    col3.metric("Mean sensor throughput", f"{100 * performance['throughput']:.2f}%")
    st.line_chart({"Best validation loss": result.history})
    arch = describe_architecture(result.architecture, result.parameters, config.dx, config.spectrum)
    st.subheader("Learned architecture")
    for i, element in enumerate(arch):
        values = ", ".join(f"{key}={value:.5g}" for key, value in element.items() if key != "operation")
        st.write(f"{i + 1}. {element['operation']}" + (f" ({values})" if values else ""))
    for message in sampling_warnings(result.architecture, result.parameters,
            config.shape, config.dx, config.spectrum, config.pad_width):
        st.warning(message)
    st.caption("This is a small, noise-free scalar simulation. Verify a promising design with finer "
               "spatial sampling and larger padding before drawing conclusions about physical performance.")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    draw_schematic(axes[0], describe_architecture(result.initial_architecture,
        result.initial_parameters, config.dx, config.spectrum), "Example initial architecture")
    draw_schematic(axes[1], arch, "Saved best architecture")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    target, output = performance["targets"][0], performance["outputs"][0]
    for ax, image, title in zip(axes, (test[0], target, output),
            ("Test object", "Target on sensor grid", "Saved winner output")):
        limits = {"vmin": 0, "vmax": max(float(target.max()), float(output.max()))} \
            if config.task == AMPLITUDE and title != "Test object" else {}
        im = ax.imshow(image, cmap="gray", **limits)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    export = {"config": asdict(config), "architecture": arch,
              "architecture_indices": result.architecture.tolist(),
              "parameters": {name: value.tolist() for name, value in zip(("z", "f", "w"), result.parameters)},
              "validation_loss": result.validation_loss, "test_loss": performance["loss"],
              "test_throughput": performance["throughput"], "history": result.history}
    st.download_button("Download design and settings", json.dumps(export, indent=2),
                       "oas_design.json", "application/json")


if __name__ == "__main__":
    main()
