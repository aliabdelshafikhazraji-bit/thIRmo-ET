"""Tests of the routed prediction (version 5.0.0) and of the broad-band branch."""
from pathlib import Path
import sys,unittest,json,tempfile,subprocess
import numpy as np
from scipy.special import voigt_profile
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
import thirmo_three as tt
from spectral_model import ThreeSpectrumPredictor
from broad_band import broad_band as BB

TEMPS=[278.15,298.15,323.15]

def broad_curves():
    """A smooth band that shifts to lower wavenumber and broadens slightly with temperature."""
    x=np.linspace(1000,1200,4001);out=[]
    for T in TEMPS:
        c=1100-0.004*(T-298.15);w=12+0.01*(T-298.15)
        out.append((x,1e-19*np.exp(-0.5*((x-c)/w)**2)*12/w))
    return out

def line_curves(model):
    x=np.linspace(1900,1940,2001);d=model.line.gaussian_coefficient;out=[]
    for T in TEMPS:
        y=np.zeros_like(x)
        for center,energy in [(1908,280),(1920,400),(1932,600)]:
            area=1e-20*(323.15/T)*np.exp(1.438776877*energy*(1/323.15-1/T))
            y+=area*voigt_profile(x-center,d*center*np.sqrt(T),.085*(296/T)**.7)
        out.append((x,y))
    return out

class RoutedPredictionTests(unittest.TestCase):
    def test_weights_match_recorded_hash(self):
        forest=BB.Forest()
        self.assertEqual(forest.sha256,forest.metadata['weights_sha256'])
        self.assertEqual(len(forest.offsets)-1,128)
    def test_broad_band_molecule_uses_broad_band_branch(self):
        r=tt.predict(broad_curves(),TEMPS,[600,900],[1],[1000,1200])
        self.assertEqual(r['metadata']['branch'],'broad_band')
        y=r['cross_section_cm2_molecule'];self.assertEqual(y.shape,(2,512))
        self.assertTrue(np.isfinite(y).all());self.assertTrue((y>=0).all())
        lo,hi=r['interval90']['low'],r['interval90']['high'];self.assertTrue(np.all(lo<=y+1e-30));self.assertTrue(np.all(y<=hi+1e-30))
    def test_line_resolved_molecule_uses_published_branch(self):
        model=ThreeSpectrumPredictor();curves=line_curves(model)
        r=tt.predict(curves,TEMPS,[800],[1],[1900,1940],probe_wavenumber_cm1=1920.013,output_fwhm_cm1=2.)
        self.assertEqual(r['metadata']['branch'],'line_resolved');self.assertIsNone(r['interval90'])
        ctx=model.prepare(curves,TEMPS,[1900,1940]);expected=model.evaluate(ctx,[800],[1],[1920.013],output_fwhm_cm1=2.)
        np.testing.assert_allclose(r['cross_section_cm2_molecule'][:,0],expected[:,0],rtol=2e-15,atol=0)
    def test_probe_equals_spectrum_value(self):
        full=tt.predict(broad_curves(),TEMPS,[750],[1],[1000,1200])
        probe=tt.predict(broad_curves(),TEMPS,[750],[1],[1000,1200],probe_wavenumber_cm1=1093.3)
        np.testing.assert_allclose(probe['cross_section_cm2_molecule'][0,0],np.interp(1093.3,full['wavenumber_cm1'],full['cross_section_cm2_molecule'][0]),rtol=1e-12)
    def test_input_order_does_not_change_prediction(self):
        c=broad_curves()
        a=tt.predict(c,TEMPS,[800],[1],[1000,1200])['cross_section_cm2_molecule']
        b=tt.predict(c[::-1],TEMPS[::-1],[800],[1],[1000,1200])['cross_section_cm2_molecule']
        np.testing.assert_array_equal(a,b)
    def test_broad_band_output_does_not_depend_on_pressure(self):
        r=tt.predict(broad_curves(),TEMPS,[800,800],[0.7,1.8],[1000,1200])['cross_section_cm2_molecule']
        np.testing.assert_array_equal(r[0],r[1])
    def test_physical_stage_conserves_band_area(self):
        g=BB.prepare(broad_curves(),TEMPS,[1000,1200],'cm2/molecule');m=BB.BroadBandPredictor()
        for T in (600.,900.):
            y=m.predict(g,T,physical_stage_only=True)
            self.assertAlmostEqual(np.trapezoid(y,g['x'])/g['area'],1.0,places=3)
    def test_invalid_conditions_are_rejected(self):
        for T,P in [(298,1),(800,.5),(800,2.1)]:
            with self.assertRaises(ValueError):tt.predict(broad_curves(),TEMPS,[T],[P],[1000,1200])
    def test_cli_writes_interval_columns_for_broad_bands(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);arguments=[]
            for i,((x,y),T) in enumerate(zip(broad_curves(),TEMPS)):
                path=root/f'cold_{i}.csv';np.savetxt(path,np.column_stack([x,y]),delimiter=',');arguments.extend(['--spectrum',str(path),str(T)])
            output=root/'prediction.csv'
            cmd=[sys.executable,str(PACKAGE/'thirmo_three.py'),*arguments,'--units','cm2-molecule','--target-temperature','800','--target-pressure','1','--context-band','1000','1200','--output',str(output)]
            process=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(process.returncode,0,process.stderr)
            data=np.loadtxt(output,delimiter=',',skiprows=1,ndmin=2);self.assertEqual(data.shape,(512,6))
            expected=tt.predict(broad_curves(),TEMPS,[800],[1],[1000,1200])['cross_section_cm2_molecule'][0]
            np.testing.assert_allclose(data[:,3],expected,rtol=1e-12)
            self.assertEqual(json.loads(output.with_suffix('.json').read_text())['branch'],'broad_band')

if __name__=='__main__':unittest.main()
