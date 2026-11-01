import unittest
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import numpy as np
from recalculate_histories import recalculate

class HistoryCalculationTests(unittest.TestCase):
    def test_known_constant_ratio(self):
        t=[0,.2,1,2];x=np.array([1.,.9,.6,.2])
        result=recalculate(t,x,2.,1.,'m2/mol')
        np.testing.assert_array_equal(result['recalculated_mole_fraction'],2*x)
        self.assertAlmostEqual(result['integrated_absolute_difference_percent'],100.)
    def test_variable_ratio_and_nonuniform_time(self):
        result=recalculate([0,1,3],[1,2,1],[1,2,1],1.,'cm2/molecule')
        self.assertAlmostEqual(result['integrated_absolute_difference_percent'],100*3/4.5)
    def test_negative_measurements_are_preserved(self):
        result=recalculate([0,1,2],[-1,1,1],2.,1.,'m2/mol')
        np.testing.assert_array_equal(result['recalculated_mole_fraction'],[-2,2,2])
        self.assertAlmostEqual(result['integrated_absolute_difference_percent'],100*2/1.5)
    def test_invalid_time_and_cross_sections_are_rejected(self):
        for times in [[0,0,1],[1,0,2],[0,np.nan,2]]:
            with self.assertRaises(ValueError):recalculate(times,[1,1,1],1.,1.,'m2/mol')
        for sigma in [0.,-1.,np.inf,[1,np.nan,1]]:
            with self.assertRaises(ValueError):recalculate([0,1,2],[1,1,1],sigma,1.,'m2/mol')
        with self.assertRaises(ValueError):recalculate([0,1,2],[-1,0,-1],1.,1.,'m2/mol')
    def test_calibration_bounds_are_inverse_and_reference_only(self):
        result=recalculate([0,1],[1,2],2.,1.,'m2/mol',reported_sigma_fraction=.06)
        np.testing.assert_allclose(result['published_uncertainty_lower'],np.array([1,2])/1.06)
        np.testing.assert_allclose(result['published_uncertainty_upper'],np.array([1,2])/.94)
        self.assertNotIn('predicted_uncertainty_lower',result)
    def test_reported_concentration_bounds(self):
        result=recalculate([0,1],[-.1,1],1.,1.,'m2/mol',reported_x_fraction=.059)
        np.testing.assert_allclose(result['published_uncertainty_lower'],[-.1059,.941])
        np.testing.assert_allclose(result['published_uncertainty_upper'],[-.0941,1.059])
    def test_cli_writes_evaluation_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'supplied.csv';output=root/'comparison.csv'
            source.write_text('time_ms,published_mole_fraction,sigma_reference,sigma_predicted\n0,0.1,4,2\n1,0.2,4,2\n2,0.1,4,2\n',encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('recalculate_histories.py')),
                '--input',str(source),'--output',str(output),'--cross-section-units','m2/mol','--reported-x-fraction','0.059']
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            metadata=json.loads(output.with_suffix('.json').read_text())
            self.assertAlmostEqual(metadata['integrated_absolute_difference_percent'],100.)
            self.assertTrue(metadata['evaluation_only'])
            self.assertFalse(metadata['model_inference_called'])
            self.assertFalse(metadata['fitting_or_normalization'])

if __name__=='__main__':unittest.main()
