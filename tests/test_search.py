import jax.numpy as jnp
import numpy as np
import pytest

from oas_core import AMPLITUDE, PHASE, SearchConfig, best_candidate, evaluate, run_search
from oas_data import split_indices


def test_dataset_splits_are_disjoint_and_reproducible():
    train = split_indices(100, 80, 'train', 42)
    val = split_indices(100, 20, 'validation', 42)
    assert set(train).isdisjoint(val)
    assert len(set(train) | set(val)) == 100
    np.testing.assert_array_equal(train, split_indices(100, 80, 'train', 42))
    assert not np.array_equal(train, split_indices(100, 80, 'train', 43))
    with pytest.raises(ValueError):
        split_indices(100, 21, 'validation')


def test_loader_uses_official_test_set(monkeypatch):
    import sys
    import types
    from PIL import Image
    from oas_data import load_cifar_batch
    calls = []
    class FakeCIFAR:
        def __init__(self, root, train, download):
            calls.append(train)
        def __len__(self):
            return 100
        def __getitem__(self, i):
            return Image.fromarray(np.full((4,4,3), i, dtype=np.uint8)), 0
    monkeypatch.setitem(sys.modules, 'torchvision.datasets', types.SimpleNamespace(CIFAR10=FakeCIFAR))
    train = load_cifar_batch(5, 8, 'train')
    val = load_cifar_batch(5, 8, 'validation')
    test = load_cifar_batch(5, 8, 'test')
    assert calls == [True, True, False]
    assert train.shape == val.shape == test.shape == (5, 8, 8)
    assert not set(train[:, 0, 0]) & set(val[:, 0, 0])


def test_winner_rejects_nonfinite_fitness():
    assert best_candidate([np.nan, 4., 1., np.inf]) == 2
    with pytest.raises(FloatingPointError):
        best_candidate([np.nan, np.inf])


@pytest.mark.parametrize('kwargs', [dict(pop_size=0), dict(num_blocks=0), dict(gd_steps=0),
    dict(generations=0), dict(pad_width=-1), dict(f1=0), dict(dx=np.nan), dict(task='unknown')])
def test_bad_config_fails_before_compilation(kwargs):
    with pytest.raises(ValueError):
        SearchConfig(**kwargs)


@pytest.mark.parametrize('task', [AMPLITUDE, PHASE])
def test_returned_winner_reproduces_its_reported_validation_loss(task):
    config = SearchConfig(shape=(8,8), pad_width=4, num_blocks=2,
        pop_size=3, gd_steps=1, generations=2, seed=3, task=task)
    images = np.random.default_rng(13).random((6,8,8), dtype=np.float32)
    progress = []
    result = run_search(config, images[:3], images[3:5],
                        lambda gen, loss: progress.append((gen, loss)))
    actual = evaluate(result, config, images[3:5])
    assert actual['loss'] == pytest.approx(result.validation_loss, rel=2e-6, abs=2e-6)
    assert len(progress) == len(result.history) == 2
    assert np.all(np.diff(result.history) <= 0)
    assert np.isfinite(evaluate(result, config, images[5:])['loss'])


def test_seed_reproduces_complete_search():
    config = SearchConfig(shape=(4,4), pad_width=2, num_blocks=1,
        pop_size=2, gd_steps=1, generations=2, seed=10)
    images = np.random.default_rng(5).random((4,4,4), dtype=np.float32)
    a = run_search(config, images[:2], images[2:])
    b = run_search(config, images[:2], images[2:])
    np.testing.assert_array_equal(a.architecture, b.architecture)
    for x,y in zip(a.parameters,b.parameters):
        np.testing.assert_array_equal(x,y)
    assert a.history == b.history


def test_population_of_one_and_one_generation_work():
    config = SearchConfig(shape=(4,4), pad_width=2, num_blocks=1,
        pop_size=1, gd_steps=1, generations=1)
    image = jnp.ones((1,4,4)) * .5
    result = run_search(config, image, image)
    assert np.isfinite(result.validation_loss)
    assert evaluate(result, config, image)['loss'] == pytest.approx(result.validation_loss)


def test_invalid_batch_fails_before_search():
    config = SearchConfig(shape=(4,4), pop_size=1, generations=1)
    with pytest.raises(ValueError):
        run_search(config, np.zeros((0,4,4)), np.ones((1,4,4)))
    with pytest.raises(ValueError):
        run_search(config, np.full((1,4,4),np.nan), np.ones((1,4,4)))
