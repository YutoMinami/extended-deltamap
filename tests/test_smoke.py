from __future__ import annotations

import subprocess
import sys
import unittest
import configparser
import importlib.util
from pathlib import Path

import numpy as np
import sympy

from extended_deltamap import (
    Covariance,
    DMatrix,
    DeltaMap,
    Templates,
    expand_to_qu,
    validate_region_masks,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
MASK_PATH = REPO_ROOT / "examples" / "files" / "mask_p06_Nside4.v2.fits"
TEST_DELTAMAP_PATH = REPO_ROOT / "examples" / "TestDeltamap.py"

_spec = importlib.util.spec_from_file_location("example_testdeltamap", TEST_DELTAMAP_PATH)
assert _spec is not None and _spec.loader is not None
example_testdeltamap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(example_testdeltamap)


class SmokeTests(unittest.TestCase):
    def test_package_import_exports_expected_symbols(self) -> None:
        import extended_deltamap

        self.assertEqual(
            extended_deltamap.__all__,
            [
                "Covariance",
                "DeltaMap",
                "DMatrix",
                "Templates",
                "expand_to_qu",
                "validate_region_masks",
            ],
        )

    def test_region_helpers_expand_and_validate_masks(self) -> None:
        mask = np.array([True, False, True])
        expanded = expand_to_qu(mask)

        self.assertTrue(
            np.array_equal(
                expanded,
                np.array([True, False, True, True, False, True]),
            )
        )

        analysis_mask = np.array([True, True, True])
        good_a = np.array([True, False, False])
        good_b = np.array([False, True, True])
        validate_region_masks([good_a, good_b], analysis_mask)

        bad_a = np.array([True, True, False])
        bad_b = np.array([False, True, True])
        with self.assertRaisesRegex(ValueError, "overlaps"):
            validate_region_masks([bad_a, bad_b], analysis_mask)

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
        self.assertEqual(dmatrix.column_masks, [None])

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
            sympy.simplify(sympy.diff(dust, params[0], params[0]) / 2),
            sympy.simplify(sympy.diff(dust, params[0], params[1])),
            sympy.simplify(sympy.diff(dust, params[1], params[1]) / 2),
        ]
        for actual_term, expected_term in zip(terms[3:], expected_second_order):
            self.assertEqual(sympy.simplify(actual_term - expected_term), 0)

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

    def test_templates_accept_region_specific_symbol_names(self) -> None:
        templates = Templates()
        synch = templates.ReturnPowerLawSynch(symbol_name="beta_s_sreg0")
        dust = templates.ReturnMBB1(
            beta_symbol_name="beta_d_dreg0",
            temperature_symbol_name="T_d1_dreg0",
        )

        synch_params = sorted(
            param.name for param in synch.free_symbols if param.name != "nu"
        )
        dust_params = sorted(
            param.name for param in dust.free_symbols if param.name != "nu"
        )

        self.assertEqual(synch_params, ["beta_s_sreg0"])
        self.assertEqual(dust_params, ["T_d1_dreg0", "beta_d_dreg0"])

    def test_dmatrix_uses_region_specific_template_symbols(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dmatrix.AddD(templates.ReturnPowerLawSynch(symbol_name="beta_s_sreg0"))
        dmatrix.AddD(templates.ReturnPowerLawSynch(symbol_name="beta_s_sreg1"))
        dmatrix.SetFreqs([40.0, 60.0, 140.0], [None, None, None])

        dmatrix.PrepareDMatrix(order=1)

        params_by_component = [
            [param.name for param in params] for params in dmatrix.d_params
        ]
        self.assertEqual(params_by_component, [["beta_s_sreg0"], ["beta_s_sreg1"]])
        self.assertEqual(dmatrix.D_matrix.shape, (3, 4))

    def test_dmatrix_expands_component_masks_to_derivative_columns(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        mask0 = np.array([True, False, True, False])
        mask1 = np.array([False, True, False, True])
        dmatrix.AddD(
            templates.ReturnPowerLawSynch(symbol_name="beta_s_sreg0"),
            region_mask=mask0,
        )
        dmatrix.AddD(
            templates.ReturnPowerLawSynch(symbol_name="beta_s_sreg1"),
            region_mask=mask1,
        )
        dmatrix.SetFreqs([40.0, 60.0, 140.0], [None, None, None])

        dmatrix.PrepareDMatrix(order=1)

        self.assertEqual(len(dmatrix.column_masks), 4)
        self.assertIs(dmatrix.column_masks[0], mask0)
        self.assertIs(dmatrix.column_masks[1], mask0)
        self.assertIs(dmatrix.column_masks[2], mask1)
        self.assertIs(dmatrix.column_masks[3], mask1)

        dmatrix.PrepareUniformDMatrix()
        self.assertEqual(len(dmatrix.column_masks), 2)
        self.assertIs(dmatrix.column_masks[0], mask0)
        self.assertIs(dmatrix.column_masks[1], mask1)

    def test_count_component_terms_matches_second_order_dust_columns(self) -> None:
        self.assertEqual(example_testdeltamap.count_component_terms(2, 0), 1)
        self.assertEqual(example_testdeltamap.count_component_terms(2, 1), 3)
        self.assertEqual(example_testdeltamap.count_component_terms(2, 2), 6)

    def test_dmatrix_prefactor_generalizes_to_third_order(self) -> None:
        templates = Templates()
        dmatrix = DMatrix()
        dust = templates.ReturnMBB1()
        params = dmatrix._sorted_params([param for param in dust.free_symbols if param.name != "nu"])

        terms = dmatrix._build_component_terms(dust, params, max_order=3)

        expected_third_order = [
            sympy.simplify(sympy.diff(dust, params[0], params[0], params[0]) / 6),
            sympy.simplify(sympy.diff(dust, params[0], params[0], params[1]) / 2),
            sympy.simplify(sympy.diff(dust, params[0], params[1], params[1]) / 2),
            sympy.simplify(sympy.diff(dust, params[1], params[1], params[1]) / 6),
        ]

        self.assertEqual(len(terms), 10)
        for actual_term, expected_term in zip(terms[6:], expected_third_order):
            self.assertEqual(sympy.simplify(actual_term - expected_term), 0)

    def test_validate_fit_setup_requires_enough_bands_for_second_order_dust(self) -> None:
        map_parser = configparser.ConfigParser()
        map_parser.read_dict(
            {
                "par": {
                    "nu": "100 119 140 166 195 235 280 337 402",
                    "fwhm": "1 1 1 1 1 1 1 1 1",
                    "noise": "1 1 1 1 1 1 1 1 1",
                }
            }
        )
        fit_parser = configparser.ConfigParser()
        fit_parser.read_dict(
            {
                "par": {
                    "nu": "100 119 140 166 195 235 280 337 402",
                    "isdust": "True",
                    "issynch": "False",
                    "fixTd": "False",
                    "params": "r T_d1 beta_d beta_s",
                    "inits": "0.5 20.1 1.5 -3.0",
                    "oname": "num0001.csv",
                    "order": "2",
                }
            }
        )

        example_testdeltamap.validate_fit_setup(
            map_parser=map_parser,
            fit_parser=fit_parser,
            nu_list=np.array([100, 119, 140, 166, 195, 235, 280, 337, 402], dtype=float),
            fwhm_list=np.ones(9),
            noise_list=np.ones(9),
            nu_list_fit=np.array([100, 119, 140, 166, 195, 235, 280, 337, 402], dtype=float),
            fit_params=["r", "T_d1", "beta_d", "beta_s"],
            fit_inits=np.array([0.5, 20.1, 1.5, -3.0], dtype=float),
        )

        with self.assertRaisesRegex(ValueError, "Not enough fitting frequencies"):
            example_testdeltamap.validate_fit_setup(
                map_parser=map_parser,
                fit_parser=fit_parser,
                nu_list=np.array([100, 119, 140, 166, 195, 235, 280, 337, 402], dtype=float),
                fwhm_list=np.ones(9),
                noise_list=np.ones(9),
                nu_list_fit=np.array([166, 195, 235, 280, 337, 402], dtype=float),
                fit_params=["r", "T_d1", "beta_d", "beta_s"],
                fit_inits=np.array([0.5, 20.1, 1.5, -3.0], dtype=float),
            )

    def test_stable_cholesky_handles_small_negative_roundoff(self) -> None:
        dmp = DeltaMap()
        nearly_pd = np.array(
            [
                [1.0, 0.9999999999995],
                [0.9999999999995, 0.9999999999989],
            ]
        )

        factor = dmp.stable_cholesky(nearly_pd, lower=True)

        reconstructed = factor @ factor.T
        self.assertEqual(factor.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(factor)))
        self.assertTrue(np.allclose(reconstructed, reconstructed.T))

    def test_masked_noise_block_applies_row_and_column_masks(self) -> None:
        dmp = DeltaMap()
        noise_block = np.array([[1.0, 2.0], [3.0, 4.0]])
        row_mask = np.array([True, False])
        column_mask = np.array([False, True])

        masked = dmp._masked_noise_block(
            noise_block,
            row_mask=row_mask,
            column_mask=column_mask,
        )

        self.assertTrue(
            np.array_equal(masked, np.array([[0.0, 2.0], [0.0, 0.0]]))
        )
        self.assertIs(dmp._masked_noise_block(noise_block), noise_block)

        with self.assertRaisesRegex(ValueError, "Region mask length"):
            dmp._masked_noise_block(noise_block, row_mask=np.array([True]))


if __name__ == "__main__":
    unittest.main()
