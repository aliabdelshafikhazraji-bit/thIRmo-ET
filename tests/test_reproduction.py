"""Verify the published aggregation and source-unit/missing-data conventions."""
import csv
import json
import unittest
from pathlib import Path
import numpy as np
from reproduction.pipeline import summarize, canonical_cross_section, smooth_record, NA, KB

ROOT=Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_published_record_aggregation(self):
        with (ROOT/'study/record_errors.csv').open(newline='',encoding='utf8') as f:
            rows=list(csv.DictReader(f))
        for r in rows:r['error_percent']=float(r['final_error_percent'])
        expected=json.loads((ROOT/'study/aggregation_expected.json').read_text())['expected']['final']
        actual=summarize(rows)
        self.assertEqual(actual['all']['n_records'],84)
        self.assertEqual(actual['HITEMP']['n_records'],39)
        self.assertEqual(actual['measurement']['n_records'],45)
        for domain in actual:
            self.assertAlmostEqual(actual[domain]['mean_error_percent'],expected[domain]['mean_error_percent'],places=11)

    def test_temperature_is_used_for_pressure_normalized_coefficient(self):
        kappa=np.array([0.,1e-7,3e-7])
        a=canonical_cross_section(kappa,'Pa-1 m-1',300)
        b=canonical_cross_section(kappa,'Pa-1 m-1',600)
        np.testing.assert_allclose(b,2*a,rtol=1e-14,atol=0)
        self.assertAlmostEqual(a[1]/(1e-7*KB*300*1e4),1.)

    def test_molar_cross_section_unit_conversion(self):
        a=canonical_cross_section([100.],'m2/mol',700)
        b=canonical_cross_section([1e6],'cm2/mol',700)
        np.testing.assert_allclose(a,b,rtol=1e-14,atol=0)
        self.assertAlmostEqual(a[0]*NA,1e6)

    def test_reference_gaps_remain_unobserved(self):
        x=np.r_[np.arange(1100.,1151.),np.arange(1170.,1221.)]
        y=np.ones(len(x))*1e-20
        grid=np.linspace(1100,1220,512)
        out,mask=smooth_record(x,y,grid)
        gap=(grid>1150)&(grid<1170)
        self.assertTrue(np.isnan(out[gap]).all())
        self.assertFalse(mask[gap].any())


if __name__=='__main__':unittest.main()
