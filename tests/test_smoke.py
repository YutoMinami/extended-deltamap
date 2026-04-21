from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import sympy

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

    def test_dmatrix_second_order_helper_uses_stable_unique_pairs(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dust = templates.ReturnMBB1()
        params = dmatrix._sorted_params(
            [param for param in dust.free_symbols if param.name != "nu"]
        )

        terms = dmatrix._build_component_terms(dust, params, max_order=2)

        self.assertEqual([param.name for param in params], ["T_d1", "beta_d"])
        self.assertEqual(len(terms), 6)
        self.assertEqual(terms[0], dust)
        expected_first_order = [
            sympy.simplify(sympy.diff(dust, params[0])),
            sympy.simplify(sympy.diff(dust, params[1])),
        ]
        self.assertEqual(terms[1:3], expected_first_order)
        expected_second_order = [
            sympy.simplify(sympy.diff(dust, params[0], params[0])),
            sympy.simplify(sympy.diff(dust, params[0], params[1])),
            sympy.simplify(sympy.diff(dust, params[1], params[1])),
        ]
        self.assertEqual(terms[3:], expected_second_order)

    def test_prepare_dmatrix_order_controls_column_count_and_dim_params(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dmatrix.AddD(templates.ReturnMBB1())
        dmatrix.SetFreqs([40.0, 140.0], [None, None])

        dmatrix.PrepareDMatrix(order=0)
        self.assertEqual(dmatrix.D_matrix.shape, (2, 1))
        self.assertEqual(dmatrix.dim_params, 1)

        dmatrix.PrepareDMatrix(order=1)
        self.assertEqual(dmatrix.D_matrix.shape, (2, 3))
        self.assertEqual(dmatrix.dim_params, 3)

        dmatrix.PrepareDMatrix(order=2)
        self.assertEqual(dmatrix.D_matrix.shape, (2, 6))
        self.assertEqual(dmatrix.dim_params, 6)

    def test_prepare_uniform_dmatrix_matches_order_zero(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dmatrix.AddD(templates.ReturnPowerLawSynch())
        dmatrix.SetFreqs([30.0, 40.0, 90.0], [None, None, None])

        dmatrix.PrepareUniformDMatrix()
        uniform_matrix = dmatrix.D_matrix
        uniform_dim = dmatrix.dim_params

        dmatrix.PrepareDMatrix(order=0)
        self.assertEqual(dmatrix.D_matrix, uniform_matrix)
        self.assertEqual(uniform_dim, dmatrix.dim_params)
        self.assertEqual(dmatrix.dim_params, 1)

    def test_prepare_dmatrix_keeps_component_block_order(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dust = templates.ReturnMBB1()
        synch = templates.ReturnPowerLawSynch()
        dmatrix.AddD(dust)
        dmatrix.AddD(synch)
        dmatrix.SetFreqs([40.0, 140.0], [None, None])

        dmatrix.PrepareDMatrix(order=1)

        dust_params = dmatrix._sorted_params(
            [param for param in dust.free_symbols if param.name != "nu"]
        )
        synch_params = dmatrix._sorted_params(
            [param for param in synch.free_symbols if param.name != "nu"]
        )
        expected_terms = (
            dmatrix._build_component_terms(dust, dust_params, max_order=1)
            + dmatrix._build_component_terms(synch, synch_params, max_order=1)
        )

        self.assertEqual(dmatrix.D_matrix_template, expected_terms)
        self.assertEqual(dmatrix.dim_params, len(expected_terms))


if __name__ == "__main__":
    unittest.main()
