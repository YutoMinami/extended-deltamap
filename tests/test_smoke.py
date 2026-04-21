from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from extended_deltamap import Covariance, DMatrix, Templates


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = REPO_ROOT / "examples" / "files" / "mask_p06_Nside4.v2.fits"


class SmokeTests(unittest.TestCase):
    def test_package_import_exports_expected_symbols(self) -> None:
        import extended_deltamap

        self.assertEqual(
            extended_deltamap.__all__,
            ["Covariance", "DeltaMap", "DMatrix", "Templates"],
        )

    def test_module_entrypoint_runs(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "extended_deltamap.deltamap"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertNotIn("NameError", result.stderr)

    def test_covariance_smoke(self) -> None:
        cov = Covariance(
            nside=4,
            lmax=8,
            maskname=str(MASK_PATH),
            pixwin=True,
            fwhm=2200.0,
            verbose=False,
        )
        cov.Initialise()

        scalar_cov = cov.ReturnCovMatrix(True)
        tensor_cov = cov.ReturnCovMatrix(False)

        self.assertEqual(scalar_cov.shape, tensor_cov.shape)
        self.assertEqual(scalar_cov.shape[0], scalar_cov.shape[1])
        self.assertTrue(np.isfinite(scalar_cov).all())
        self.assertTrue(np.isfinite(tensor_cov).all())

    def test_dmatrix_rebuilds_without_accumulating_columns(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dmatrix.AddD(templates.ReturnPowerLawSynch())
        dmatrix.SetFreqs([30.0, 40.0, 90.0], [None, None, None])

        dmatrix.PrepareDMatrix()
        first_shape = dmatrix.D_matrix.shape
        dmatrix.PrepareDMatrix()
        second_shape = dmatrix.D_matrix.shape
        dmatrix.PrepareUniformDMatrix()
        uniform_shape = dmatrix.D_matrix.shape

        self.assertEqual(first_shape, (3, 2))
        self.assertEqual(second_shape, first_shape)
        self.assertEqual(uniform_shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
