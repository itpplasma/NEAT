import logging
import os
import unittest
from unittest.mock import Mock, patch

import numpy as np
from numpy.testing import assert_almost_equal

from neat.fields import Simple, Stellna, StellnaQS, Vmec
from neat.simple_classification import SimpleLossResult

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NEATtests(unittest.TestCase):
    def test_StellnaQS(self):
        """
        Test that we can obtain qs fields from pyQSC
        using several methods
        """
        assert_almost_equal(
            StellnaQS.from_paper(1).iota, -0.4204733518104154, decimal=10
        )
        assert_almost_equal(
            StellnaQS(
                rc=[1, 0.155, 0.0102],
                zs=[0, 0.154, 0.0111],
                nfp=2,
                etabar=0.64,
                order="r3",
                B2c=-0.00322,
            ).iota,
            -0.4204733518104154,
            decimal=10,
        )
        assert_almost_equal(
            StellnaQS.from_paper("r1 section 5.3").iota, 0.3111813731231253, decimal=10
        )
        assert_almost_equal(
            StellnaQS(
                rc=[1, 0.042],
                zs=[0, -0.042],
                zc=[0, -0.025],
                nfp=3,
                etabar=-1.1,
                sigma0=-0.6,
            ).iota,
            0.3111813731231253,
            decimal=10,
        )

    def test_Stellna(self):
        """
        Test that we can obtain qi fields from pyQIC
        using several methods
        """
        assert_almost_equal(
            Stellna.from_paper("QI").iota, 0.7166463779543341, decimal=10
        )

        assert_almost_equal(
            Stellna(
                rc=[1, 0.155, 0.0102],
                zs=[0, 0.154, 0.0111],
                nfp=2,
                etabar=0.1,
                order="r3",
                B2c=-0.01,
                nphi=251,
            ).iota,
            -0.018692578813516082,
            decimal=10,
        )
        self.assertAlmostEqual(
            StellnaQS.from_paper(1).gyronimo_parameters()[0], 1.0470998216534495
        )
        self.assertAlmostEqual(Stellna.from_paper(1).gyronimo_parameters()[0], 2)

    def setUp(self):
        self.wout_filename = os.path.join(
            os.path.dirname(__file__), "inputs", "wout_ARIESCS.nc"
        )
        self.simple_object = Simple(
            wout_filename=self.wout_filename,
            B_scale=1.0,
            Aminor_scale=1.0,
            multharm=3,
        )
        self.vmec = Vmec(self.wout_filename)

    def test_stellna_gyronimo_parameters(self):
        field = Stellna.from_paper(1)
        result = field.gyronimo_parameters()

        self.assertIsInstance(result[0], int)
        self.assertEqual(result[0], int(field.nfp))

        varphi = result[7]
        self.assertEqual(len(varphi), len(field.varphi) + 1)
        self.assertAlmostEqual(varphi[-1], 2 * np.pi / field.nfp + varphi[0])

    def test_simple_run_loss_populates_params(self):
        from pathlib import Path

        fake_confined = np.array(
            [
                [0.0, 0.5, 0.5, 2.0],
                [1.0e-3, 0.5, 0.25, 2.0],
            ],
            dtype=float,
        )
        fake_times_lost = np.array(
            [
                [1, 1.0e-3, -0.1, 0.5, 0.01, 0, 0, 0, 0, 0],
                [2, 5.0e-4, 0.2, 0.5, 0.02, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )
        fake_loss_fraction_vs_time = 1.0 - (fake_confined[:, 1] + fake_confined[:, 2])

        with patch("neat.fields.run_simple_loss") as mocked:
            mocked.return_value = SimpleLossResult(
                workdir=Path("/tmp/neat_test_simple_workdir"),
                confined_fraction=fake_confined,
                times_lost=fake_times_lost,
                loss_fraction=float(fake_loss_fraction_vs_time[-1]),
                loss_fraction_vs_time=fake_loss_fraction_vs_time,
            )

            params = self.simple_object.run_loss(
                tfinal=1.0e-3,
                nsamples=2,
                nparticles=2,
                npoiper=123,
                npoiper2=321,
                nper=999,
                notrace_passing=1,
                deterministic=True,
                timeout_s=1.0,
            )

            self.assertIsNotNone(self.simple_object.params)
            self.assertEqual(params.trace_time, 1.0e-3)
            np.testing.assert_allclose(params.time, fake_confined[:, 0])
            np.testing.assert_allclose(params.confpart_pass, fake_confined[:, 1])
            np.testing.assert_allclose(params.confpart_trap, fake_confined[:, 2])
            np.testing.assert_allclose(params.times_lost, fake_times_lost[:, 1])
            np.testing.assert_allclose(params.perp_inv, fake_times_lost[:, 4])

            _, kwargs = mocked.call_args
            cfg = kwargs["config"]
            self.assertEqual(cfg["ntestpart"], 2)
            self.assertEqual(cfg["ntimstep"], 2)
            self.assertEqual(cfg["trace_time"], 1.0e-3)
            self.assertEqual(cfg["multharm"], 3)
            self.assertEqual(cfg["ns_s"], 3)
            self.assertEqual(cfg["ns_tp"], 3)
            self.assertEqual(cfg["vmec_B_scale"], 1.0)
            self.assertEqual(cfg["vmec_RZ_scale"], 1.0)
            self.assertEqual(cfg["notrace_passing"], 1)
            self.assertTrue(cfg["deterministic"])

    def tearDown(self):
        pass

    def test_init(self):
        self.assertEqual(self.vmec.near_axis, False)
        self.assertEqual(self.vmec.wout_filename, self.wout_filename)

    def test_vmec_gyronimo_parameters(self):
        expected_parameters = [self.wout_filename]
        self.assertEqual(self.vmec.gyronimo_parameters(), expected_parameters)


if __name__ == "__main__":
    unittest.main()
