import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thirmo_three import ThreeSpectrumPredictor
from broad_model import frozen_core as core
from cold_router import route_probe,route_cold_spectra

class ThreeInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=ThreeSpectrumPredictor()
        cls.x=np.linspace(1000,1100,5001)
        cls.ts=[278.15,298.15,323.15]
        cls.spectra=[(cls.x,(1-.002*(T-278.15))*1e-20*np.exp(-.5*((cls.x-1050)/8)**2)) for T in cls.ts]
    def call(self,spectra=None,ts=None,**kwargs):
        return self.model.predict(self.spectra if spectra is None else spectra,self.ts if ts is None else ts,
            [700],1,[1000,1100],**kwargs)
    def test_exactly_three(self):
        for n in [1,2,4]:
            with self.assertRaises(ValueError):self.call(self.spectra[:n] if n<3 else self.spectra+[self.spectra[0]],self.ts[:n] if n<3 else self.ts+[330])
    def test_temperatures_are_measured_distinct_near_ambient(self):
        for ts in [[278,278,323],[270,298,323],[278,298,351]]:
            with self.assertRaises(ValueError):self.call(ts=ts)
    def test_full_band_is_original_broad_prediction(self):
        actual=self.call()
        direct=self.model.broad.predict(self.spectra,self.ts,[700],[1000,1100],units='cross-section')
        self.assertEqual(actual['cross_section_cm2_molecule'].shape,(1,512))
        np.testing.assert_array_equal(actual['cross_section_cm2_molecule'],direct['cross_section_cm2_molecule'])
    def test_pnnl_conversion_and_temperature_order(self):
        pnnl=[(x,y*core.PNNL_COLUMN/np.log(10)) for x,y in self.spectra]
        a=self.call();b=self.call(pnnl,units='pnnl')
        np.testing.assert_allclose(a['cross_section_cm2_molecule'],b['cross_section_cm2_molecule'],rtol=1e-12,atol=1e-35)
        perm=[2,0,1]
        c=self.call([self.spectra[i] for i in perm],[self.ts[i] for i in perm])
        np.testing.assert_array_equal(a['cross_section_cm2_molecule'],c['cross_section_cm2_molecule'])
    def test_broad_probe_is_envelope_interpolation(self):
        band=self.call();probe=self.call(probe_wavenumber_cm1=1050)
        self.assertEqual(probe['metadata']['branch'],'broad_band_envelope')
        self.assertAlmostEqual(probe['cross_section_cm2_molecule'][0,0]/np.interp(1050,band['wavenumber_cm1'],band['cross_section_cm2_molecule'][0]),1,places=14)
    def test_mixed_band_uses_local_not_global_gate(self):
        curves=[]
        for T in self.ts:
            y=2e-20*np.exp(-.5*((self.x-1035)/5)**2)+8e-21/(1+((self.x-1070)/.055)**2)
            curves.append((self.x,y,T))
        self.assertFalse(route_cold_spectra(curves,[1000,1100])['passed'])
        self.assertTrue(route_probe(curves,[1000,1100],1070)['passed'])
        altered=list(curves)
        altered[0]=(self.x,2e-20*np.exp(-.5*((self.x-1035)/5)**2)+8e-21/(1+((self.x-1070)/1.5)**2),self.ts[0])
        self.assertFalse(route_probe(altered,[1000,1100],1070)['passed'])
    def test_no_target_molecular_identity_arguments(self):
        names=inspect.signature(self.model.predict).parameters
        self.assertFalse({'species','molecule','mass','mass_amu','formula'}&set(names))
    def test_invalid_probe_and_pressure(self):
        with self.assertRaises(ValueError):self.call(probe_wavenumber_cm1=900)
        with self.assertRaises(ValueError):self.model.predict(self.spectra,self.ts,[700],.5,[1000,1100])
    def test_cli_roundtrip_overwrite_and_two_input_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            command=[sys.executable,str(Path(__file__).resolve().parents[1]/'thirmo_three.py')]
            spectra_args=[]
            for i,((x,y),T) in enumerate(zip(self.spectra,self.ts)):
                path=root/f'input_{i}.csv'
                np.savetxt(path,np.column_stack([x,y]),delimiter=',',header='wavenumber,cross_section',comments='')
                spectra_args.extend(['--spectrum',str(path),str(T)])
            options=['--units','cm2-molecule','--target-temperature','700','--target-pressure','1',
                '--context-band','1000','1100','--probe','1050','--output',str(root/'result.csv')]
            run=subprocess.run(command+spectra_args+options,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            meta=json.loads((root/'result.json').read_text())
            self.assertEqual(meta['branch'],'broad_band_envelope')
            self.assertEqual(meta['input_count'],3)
            again=subprocess.run(command+spectra_args+options+['--overwrite'],capture_output=True,text=True)
            self.assertEqual(again.returncode,0,again.stderr)
            invalid=subprocess.run(command+spectra_args[:6]+options+['--overwrite'],capture_output=True,text=True)
            self.assertNotEqual(invalid.returncode,0)
            self.assertIn('exactly three',invalid.stderr)

if __name__=='__main__':unittest.main()
