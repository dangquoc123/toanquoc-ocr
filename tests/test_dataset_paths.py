"""Regression: manifest-relative image paths (the first real Kaggle-run bug).

``LineRecognitionDataset`` must resolve relative image paths against the
directory that CONTAINS the manifest, not the process cwd — otherwise training
from ``/kaggle/working/vnocr-repo`` with data in ``/kaggle/working/synth``
fails with ``FileNotFoundError: images/0005155.png``.

Pure stdlib (dataset construction does not import torch).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vnocr.recognize.dataset import LineRecognitionDataset  # noqa: E402


def _make_manifest(tmp: Path) -> Path:
    (tmp / "images").mkdir(parents=True)
    (tmp / "images" / "0000001.png").write_bytes(b"fake")
    manifest = tmp / "labels.txt"
    manifest.write_text("images/0000001.png\txin chào\n", encoding="utf-8")
    return manifest


def test_root_defaults_to_manifest_dir():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        manifest = _make_manifest(tmp)
        old_cwd = os.getcwd()
        os.chdir("/")  # cwd deliberately far away from the data
        try:
            ds = LineRecognitionDataset(str(manifest))
            resolved = ds.root / ds.samples[0].image_path
            assert resolved.exists(), resolved
            assert ds.root == manifest.resolve().parent
        finally:
            os.chdir(old_cwd)


def test_explicit_root_still_wins():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        manifest = _make_manifest(tmp)
        ds = LineRecognitionDataset(str(manifest), root=str(tmp))
        assert (ds.root / ds.samples[0].image_path).exists()


def test_absolute_paths_in_manifest_untouched():
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_manifest(tmp)
        img = tmp / "images" / "0000001.png"
        manifest2 = tmp / "abs.txt"
        manifest2.write_text(f"{img}\txin chào\n", encoding="utf-8")
        ds = LineRecognitionDataset(str(manifest2))
        # pathlib: root / absolute  ->  absolute
        assert (ds.root / ds.samples[0].image_path) == img


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"  PASS  {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
