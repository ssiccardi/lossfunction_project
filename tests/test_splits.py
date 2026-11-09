"""Unit tests for subject-level and window-level splitting."""
import numpy as np
import pytest

from lossfunction_project.data.splits import (
    ArrayBundle,
    SubjectAssignment,
    apply_subject_level_split,
    apply_window_level_split,
    find_assignment_config,
    shuffle_bundle_labels,
)
from lossfunction_project.data.windowing import SubjectWindows


def _make_subject_windows(subject: str, n_windows: int, seed: int) -> SubjectWindows:
    rng = np.random.default_rng(seed)
    return SubjectWindows(
        subject=subject,
        X=rng.standard_normal((n_windows, 512)).astype(np.float32),
        y=rng.uniform(60, 100, n_windows).astype(np.float32),
        window_start_s=np.arange(n_windows, dtype=np.float32) * 2.0,
        label_index=np.arange(n_windows, dtype=np.int64),
    )


@pytest.fixture
def all_windows():
    return {
        s: _make_subject_windows(s, n_windows=20 + i, seed=i)
        for i, s in enumerate(["S1", "S2", "S3", "S4", "S5"])
    }


def test_assignment_rejects_duplicate_subject_roles():
    assignment = SubjectAssignment(
        name="bad", train_subjects=["S1", "S2"], val_subjects=["S2"], test_subjects=["S3"]
    )
    with pytest.raises(ValueError):
        assignment.validate()


def test_subject_level_split_no_subject_overlap(all_windows):
    assignment = SubjectAssignment(
        name="default", train_subjects=["S1", "S2", "S3"], val_subjects=["S4"], test_subjects=["S5"]
    )
    split = apply_subject_level_split(all_windows, assignment)

    train_subs = set(split.train.subject_id.tolist())
    val_subs = set(split.val.subject_id.tolist())
    test_subs = set(split.test.subject_id.tolist())

    assert train_subs == {"S1", "S2", "S3"}
    assert val_subs == {"S4"}
    assert test_subs == {"S5"}
    assert train_subs.isdisjoint(val_subs)
    assert train_subs.isdisjoint(test_subs)
    assert val_subs.isdisjoint(test_subs)


def test_subject_level_split_preserves_all_windows(all_windows):
    assignment = SubjectAssignment(
        name="default", train_subjects=["S1", "S2", "S3"], val_subjects=["S4"], test_subjects=["S5"]
    )
    split = apply_subject_level_split(all_windows, assignment)
    total = len(split.train) + len(split.val) + len(split.test)
    expected_total = sum(len(w.y) for w in all_windows.values())
    assert total == expected_total


def test_window_level_split_covers_all_windows_once(all_windows):
    split = apply_window_level_split(all_windows, train_frac=0.7, val_frac=0.15, seed=42)
    total = len(split.train) + len(split.val) + len(split.test)
    expected_total = sum(len(w.y) for w in all_windows.values())
    assert total == expected_total

    # Reconstruct (subject, label_index) pairs; must be unique across the whole pool
    # (i.e. every window appears in exactly one split).
    def pairs(bundle):
        return set(zip(bundle.subject_id.tolist(), bundle.label_index.tolist()))

    train_pairs = pairs(split.train)
    val_pairs = pairs(split.val)
    test_pairs = pairs(split.test)

    assert train_pairs.isdisjoint(val_pairs)
    assert train_pairs.isdisjoint(test_pairs)
    assert val_pairs.isdisjoint(test_pairs)
    assert len(train_pairs) + len(val_pairs) + len(test_pairs) == expected_total


def test_window_level_split_is_reproducible_given_seed(all_windows):
    split_a = apply_window_level_split(all_windows, seed=123)
    split_b = apply_window_level_split(all_windows, seed=123)
    np.testing.assert_array_equal(split_a.train.label_index, split_b.train.label_index)
    np.testing.assert_array_equal(split_a.train.subject_id, split_b.train.subject_id)


def test_window_level_split_mixes_subjects(all_windows):
    """Unlike the subject-level split, the window-level train set should normally
    contain windows from more than one subject (this is the point being illustrated:
    subject identity is not respected)."""
    split = apply_window_level_split(all_windows, seed=42)
    train_subjects = set(split.train.subject_id.tolist())
    assert len(train_subjects) > 1


def test_find_assignment_config_matches_by_internal_name_not_filename():
    """configs/subject_assignments/assignment_default.yaml has internal name:
    'default'; lookups must resolve by that internal name, not by filename stem.
    """
    path = find_assignment_config("default")
    assert path.name == "assignment_default.yaml"


def test_find_assignment_config_raises_for_unknown_name():
    with pytest.raises(FileNotFoundError):
        find_assignment_config("does_not_exist")


def _make_bundle(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return ArrayBundle(
        X=rng.standard_normal((n, 512)).astype(np.float32),
        y=rng.uniform(60, 100, n).astype(np.float32),
        subject_id=np.full(n, "S1", dtype=object),
        window_start_s=np.arange(n, dtype=np.float32) * 2.0,
        label_index=np.arange(n, dtype=np.int64),
    )


def test_shuffle_bundle_labels_preserves_X_and_metadata():
    bundle = _make_bundle()
    shuffled = shuffle_bundle_labels(bundle, seed=1)
    np.testing.assert_array_equal(shuffled.X, bundle.X)
    np.testing.assert_array_equal(shuffled.subject_id, bundle.subject_id)
    np.testing.assert_array_equal(shuffled.window_start_s, bundle.window_start_s)
    np.testing.assert_array_equal(shuffled.label_index, bundle.label_index)


def test_shuffle_bundle_labels_permutes_y_values_only():
    bundle = _make_bundle()
    shuffled = shuffle_bundle_labels(bundle, seed=1)
    # Same multiset of values, different order (overwhelmingly likely with n=2000).
    np.testing.assert_allclose(np.sort(shuffled.y), np.sort(bundle.y))
    assert not np.array_equal(shuffled.y, bundle.y)


def test_shuffle_bundle_labels_preserves_mean_and_std():
    bundle = _make_bundle()
    shuffled = shuffle_bundle_labels(bundle, seed=1)
    assert shuffled.y.mean() == pytest.approx(bundle.y.mean())
    assert shuffled.y.std() == pytest.approx(bundle.y.std())


def test_shuffle_bundle_labels_reproducible_given_seed():
    bundle = _make_bundle()
    shuffled_a = shuffle_bundle_labels(bundle, seed=7)
    shuffled_b = shuffle_bundle_labels(bundle, seed=7)
    np.testing.assert_array_equal(shuffled_a.y, shuffled_b.y)

