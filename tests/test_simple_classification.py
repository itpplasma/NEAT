import tempfile
import unittest
from pathlib import Path

import numpy as np

from neat.simple_classification import (
    compute_classification_metric,
    default_fast_classification_config,
    parse_class_parts,
    parse_times_lost,
    write_simple_in,
)


class SimpleClassificationTests(unittest.TestCase):
    def test_write_simple_in(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "simple.in"
            cfg = default_fast_classification_config(
                ntestpart=8, trace_time_s=1e-4, class_plot=True, tcut_s=1e-4
            )
            cfg["netcdffile"] = "wout.nc"
            write_simple_in(path, cfg)
            text = path.read_text(encoding="utf-8")
            self.assertIn("&config", text)
            self.assertIn("fast_class = .True.", text)
            self.assertIn("class_plot = .True.", text)
            self.assertIn("tcut = 0.0001", text)
            self.assertIn("multharm = 3", text)
            self.assertIn("nturns = 8", text)
            self.assertIn("netcdffile = 'wout.nc'", text)
            self.assertTrue(text.strip().endswith("/"))

    def test_parse_and_metric(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            workdir = Path(td)

            # 4 particles:
            # - 2 trapped (trap_par > 0), 2 passing (trap_par <= 0)
            # Trapped classifications:
            # - particle 1: ijpar=1, ideal=1
            # - particle 2: ijpar=2, ideal=2
            class_parts = np.array(
                [
                    [1, 0.6, 0.1, 1, 1, 0],
                    [2, 0.6, 0.1, 2, 2, 0],
                    [3, 0.6, 0.1, 1, 0, 0],
                    [4, 0.6, 0.1, 0, 0, 0],
                ],
                dtype=float,
            )
            times_lost = np.array(
                [
                    [1, 0.01, 0.5, 0.6, 0.1, 0, 0, 0, 0, 0],
                    [2, 0.01, 0.2, 0.6, 0.1, 0, 0, 0, 0, 0],
                    [3, 0.01, -0.1, 0.6, 0.1, 0, 0, 0, 0, 0],
                    [4, 0.01, 0.0, 0.6, 0.1, 0, 0, 0, 0, 0],
                ],
                dtype=float,
            )

            np.savetxt(workdir / "class_parts.dat", class_parts)
            np.savetxt(workdir / "times_lost.dat", times_lost)

            # Prompt loss markers: 1 lost passing + 1 lost trapped
            (workdir / "fort.10001").write_text("0 0 0\n", encoding="utf-8")
            (workdir / "fort.10002").write_text("0 0 0\n", encoding="utf-8")

            parsed_class = parse_class_parts(workdir / "class_parts.dat")
            parsed_times = parse_times_lost(workdir / "times_lost.dat")
            score, counts = compute_classification_metric(
                class_parts=parsed_class,
                times_lost=parsed_times,
                workdir=workdir,
                w_good=1.0,
                w_prompt=1.0,
            )

            self.assertEqual(counts.n_particles, 4)
            self.assertEqual(counts.n_trapped, 2)
            self.assertEqual(counts.n_passing, 2)
            self.assertEqual(counts.n_prompt_lost, 2)

            self.assertEqual(counts.n_trapped_classified_ideal, 1)
            self.assertEqual(counts.n_trapped_classified_nonideal, 1)
            self.assertEqual(counts.n_trapped_classified_jpar_good, 1)
            self.assertEqual(counts.n_trapped_classified_jpar_bad, 1)

            self.assertAlmostEqual(counts.ideal_fraction_trapped, 0.5)
            self.assertAlmostEqual(counts.jpar_good_fraction_trapped, 0.5)
            self.assertAlmostEqual(counts.good_fraction_trapped, 0.5)
            self.assertAlmostEqual(counts.prompt_loss_fraction, 0.5)

            self.assertAlmostEqual(score, 0.5 - 0.5)


if __name__ == "__main__":
    unittest.main()
