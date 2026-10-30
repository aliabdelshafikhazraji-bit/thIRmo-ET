import unittest
import tempfile
from pathlib import Path
import numpy as np
from thirmo_et import N, PNNL_COLUMN, NumericForest, make_features, native_smooth, to_cross_section, predict_spectra, read_spectrum, main


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.forest = NumericForest()
        cls.x = np.linspace(1150, 1350, 1001)
        cls.y = 1e-19 * np.exp(-0.5 * ((cls.x - 1240) / 20) ** 2)

    def test_unit_conversion(self):
        sigma = np.array([0, 1e-20, 2e-19])
        column = 2e17
        np.testing.assert_allclose(to_cross_section(sigma * column, "absorbance-ln", column), sigma)
        np.testing.assert_allclose(to_cross_section(sigma * column / np.log(10), "absorbance-decadic", column), sigma)
        np.testing.assert_allclose(to_cross_section(sigma * PNNL_COLUMN / np.log(10), "pnnl"), sigma)
        with self.assertRaises(ValueError):
            to_cross_section(sigma, "absorbance-ln")

    def test_missing_band_gap_is_rejected(self):
        x = np.r_[np.arange(1150, 1200), np.arange(1210, 1250)]
        with self.assertRaises(ValueError):
            native_smooth(x, np.ones_like(x), np.linspace(1150, 1249, N))

    def test_unsampled_gap_is_rejected(self):
        x = np.r_[np.arange(100, 105), np.arange(111, 501)]
        with self.assertRaises(ValueError):
            native_smooth(x, np.ones_like(x), np.array([100, 200, 500]))

    def test_inline_comment_keeps_first_numeric_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.txt"
            source.write_text("1150 1e-20 # first measured point, retained\n1151 2e-20\n1152 3e-20\n")
            x, y = read_spectrum(source)
            self.assertEqual(len(x), 3)
            self.assertEqual(x[0], 1150)
            self.assertEqual(y[0], 1e-20)

    def test_single_input_preserves_observation_count(self):
        x = np.linspace(1150, 1350, N)
        p = (1e-19 * np.exp(-0.5 * ((x - 1240) / 20) ** 2))[None, :]
        f, _, _, slope, names = make_features(p, [298.15], x)
        self.assertTrue(np.all(f[:, names.index("input_count")] == 1))
        self.assertTrue(np.all(slope == 0))
        np.testing.assert_array_equal(f[:, :9], f[:, 9:18])
        np.testing.assert_array_equal(f[:, :9], f[:, 18:27])

    def test_unit_equivalent_end_to_end_predictions(self):
        col = 2e17
        a = predict_spectra([(self.x, self.y)], [298.15], [673.15], forest=self.forest)
        b = predict_spectra([(self.x, self.y * col / np.log(10))], [298.15], [673.15],
                            units="absorbance-decadic", input_columns=[col], forest=self.forest)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-32)
        self.assertEqual(a[1].shape, (1, N))
        self.assertTrue(np.all(np.isfinite(a[1])) and np.all(a[1] >= 0))
        self.assertTrue(any("SINGLE_INPUT" in s for s in a[2]["warnings"]))

    def test_input_order_does_not_change_prediction(self):
        a = predict_spectra([(self.x, self.y), (self.x, self.y * 0.95)], [298.15, 323.15], [773.15], forest=self.forest)
        b = predict_spectra([(self.x, self.y * 0.95), (self.x, self.y)], [323.15, 298.15], [773.15], forest=self.forest)
        np.testing.assert_array_equal(a[1], b[1])

    def test_columns_follow_input_temperature_sorting(self):
        cold_col, warm_col = 2e17, 3e17
        a = predict_spectra([(self.x, self.y), (self.x, self.y * 0.95)], [298.15, 323.15], [773.15], forest=self.forest)
        b = predict_spectra([(self.x, self.y * 0.95 * warm_col), (self.x, self.y * cold_col)], [323.15, 298.15], [773.15],
                            units="absorbance-ln", input_columns=[warm_col, cold_col], forest=self.forest)
        np.testing.assert_allclose(a[1], b[1], rtol=1e-12, atol=1e-32)
        self.assertEqual(b[2]["input_column_density_molecules_per_cm2"], [cold_col, warm_col])

    def test_extrapolation_requires_hotter_target(self):
        with self.assertRaises(ValueError):
            predict_spectra([(self.x, self.y)], [298.15], [250], forest=self.forest)

    def test_cli_reads_header_and_writes_unit_labeled_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / "input.csv", Path(tmp) / "hot.csv"
            np.savetxt(source, np.column_stack([self.x, self.y]), delimiter=",", header="wavenumber,sigma", comments="")
            main(["--spectrum", str(source), "298.15", "--target-temperature", "673.15", "773.15",
                  "--units", "cross-section", "--window", "1150", "1350", "--output-column-density", "2e17",
                  "--output", str(output)])
            data = np.genfromtxt(output, delimiter=",", names=True)
            self.assertEqual(len(data), 2 * N)
            self.assertTrue(output.with_suffix(".json").exists())
            np.testing.assert_allclose(data["absorbance_ln"], data["cross_section_cm2_per_molecule"] * 2e17, rtol=2e-11)


if __name__ == "__main__":
    unittest.main()
