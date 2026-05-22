from __future__ import annotations

import subprocess
import sys
import time

import pytest

from pipeline.cancellation import PipelineCancelled
from pipeline.subprocess_utils import run_subprocess_cancellable


class TestRunSubprocessCancellable:
    def test_returns_completed_process_on_success(self):
        r = run_subprocess_cancellable([sys.executable, "-c", "print('ok')"])
        assert isinstance(r, subprocess.CompletedProcess)
        assert r.returncode == 0
        assert "ok" in r.stdout

    def test_no_cancel_runs_to_completion(self):
        # cancel renvoie toujours False -> la commande va jusqu'au bout
        r = run_subprocess_cancellable(
            [sys.executable, "-c", "print('done')"],
            cancel=lambda: False,
        )
        assert r is not None
        assert r.returncode == 0
        assert "done" in r.stdout

    def test_cancel_raises_pipeline_cancelled_quickly(self):
        # Process long; cancel renvoie True immédiatement -> doit lever vite
        start = time.time()
        with pytest.raises(PipelineCancelled):
            run_subprocess_cancellable(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cancel=lambda: True,
                poll_interval_s=0.1,
            )
        elapsed = time.time() - start
        # Interruption quasi immédiate, bien avant les 30 s du sommeil
        assert elapsed < 10

    def test_cancel_removes_partial_output(self, tmp_path):
        out = tmp_path / "partial.bin"
        out.write_bytes(b"partial")  # simule un fichier de sortie partiel
        with pytest.raises(PipelineCancelled):
            run_subprocess_cancellable(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                cancel=lambda: True,
                poll_interval_s=0.1,
                output_path=out,
            )
        assert not out.exists()

    def test_nonzero_returncode_is_reported(self):
        r = run_subprocess_cancellable([sys.executable, "-c", "import sys; sys.exit(3)"])
        assert r is not None
        assert r.returncode == 3

    def test_timeout_raises(self):
        with pytest.raises(subprocess.TimeoutExpired):
            run_subprocess_cancellable(
                [sys.executable, "-c", "import time; time.sleep(30)"],
                poll_interval_s=0.1,
                timeout_s=0.5,
            )
