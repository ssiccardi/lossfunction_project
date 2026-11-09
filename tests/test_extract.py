"""Round-trip tests for the extraction save/load logic.

Uses small synthetic arrays rather than real PPG-DaLiA data, since the raw .pkl
files are large, external, and not available in CI/other machines.
"""
import numpy as np

from lossfunction_project.data.extract import SubjectArrays, load_subject_npz, save_subject_npz


def test_save_load_roundtrip(tmp_path):
    original = SubjectArrays(
        subject="S_test",
        bvp=np.arange(100, dtype=np.float32),
        label=np.linspace(60, 90, 10, dtype=np.float32),
        activity=np.zeros(50, dtype=np.float32),
    )

    save_subject_npz(original, processed_dir=tmp_path)
    loaded = load_subject_npz("S_test", processed_dir=tmp_path)

    assert loaded.subject == original.subject
    np.testing.assert_array_equal(loaded.bvp, original.bvp)
    np.testing.assert_array_equal(loaded.label, original.label)
    np.testing.assert_array_equal(loaded.activity, original.activity)
    assert loaded.fs_bvp == original.fs_bvp
    assert loaded.fs_activity == original.fs_activity
    assert loaded.label_window_s == original.label_window_s
    assert loaded.label_shift_s == original.label_shift_s
