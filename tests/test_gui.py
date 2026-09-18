"""One complete UI run with local images, real search, and no dataset download."""

import numpy as np
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


def test_gui_runs_search_and_retains_saved_result(monkeypatch):
    import gui
    def local_images(batch_size, img_size, split, seed):
        offsets = {"train": 0, "validation": 1, "test": 2}
        return np.random.default_rng(seed + offsets[split]).random(
            (batch_size, img_size, img_size), dtype=np.float32)
    monkeypatch.setattr(gui, "cached_cifar", local_images)
    app = AppTest.from_string("import gui\ngui.main()").run(timeout=30)
    assert not app.exception
    values = {"Number of elements": 2, "Population size": 2,
              "Gradient steps per generation": 1, "Working-grid padding (pixels per side)": 8}
    for widget in app.number_input:
        if widget.label in values:
            widget.set_value(values[widget.label])
    app.slider[0].set_value(1)
    app.button[0].click().run(timeout=60)
    assert not app.exception
    assert len(app.metric) == 3
    config, result, _, performance = app.session_state["oas_result"]
    assert np.isfinite(result.validation_loss) and np.isfinite(performance["loss"])
    app.run(timeout=30)
    assert not app.exception
    assert len(app.metric) == 3
