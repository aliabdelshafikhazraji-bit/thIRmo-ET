from pathlib import Path
import sys,unittest,inspect,json
import numpy as np
from scipy.special import voigt_profile
PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE))
from spectral_model import ThreeSpectrumPredictor
from line_model.cold_line_representation import profile,add_cold_thermal_parameters,predict_diatomic_line

class CommonPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=ThreeSpectrumPredictor();cls.temps=[278.15,298.15,323.15]
        x=np.linspace(1900,1940,2001);d=cls.model.line.gaussian_coefficient
        curves=[]
        for T in cls.temps:
            y=np.zeros_like(x)
            for center,energy in [(1908,280),(1920,400),(1932,600)]:
                area=1e-20*(323.15/T)*np.exp(1.438776877*energy*(1/323.15-1/T))
                y+=area*voigt_profile(x-center,d*center*np.sqrt(T),.085*(296/T)**.7)
            curves.append((x,y))
        cls.curves=curves;cls.context=cls.model.prepare(curves,cls.temps,[1900,1940])
    def test_three_actual_input_requirement(self):
        for curves,ts in [(self.curves[:2],self.temps[:2]),(self.curves,[298.15,298.15,323.15])]:
            with self.assertRaises(ValueError):self.model.prepare(curves,ts,[1900,1940])
    def test_no_molecular_property_argument(self):
        params=inspect.signature(self.model.predict).parameters
        for name in ['species','mass','formula','molecular_identity','hot_reference']:self.assertNotIn(name,params)
        self.assertNotIn('mass_amu',self.context['line_records'][0])
    def test_cold_only_selection_is_nonempty(self):
        self.assertEqual(len(self.context['line_records']),3)
        np.testing.assert_allclose([r['center_cm1'] for r in self.context['line_records']],[1908,1920,1932],atol=.01)
    def test_same_value_inside_spectrum_or_as_probe(self):
        Ts=[500.,1200.];Ps=[1.,1.5];probe=1920.013
        single=self.model.evaluate(self.context,Ts,Ps,[probe])[:,0]
        grid=self.model.evaluate(self.context,Ts,Ps,[1900,1910,probe,1930,1940])[:,2]
        np.testing.assert_allclose(single,grid,rtol=2e-15,atol=0)
    def test_input_order_does_not_change_prediction(self):
        a=self.model.predict(self.curves,self.temps,[1000],[1],[1900,1940],probe_wavenumber_cm1=1920)
        b=self.model.predict(self.curves[::-1],self.temps[::-1],[1000],[1],[1900,1940],probe_wavenumber_cm1=1920)
        np.testing.assert_array_equal(a['cross_section_cm2_molecule'],b['cross_section_cm2_molecule'])
    def test_pressure_affects_resolved_shape_only(self):
        a=self.model.evaluate(self.context,[800,800],[1,1.5],[1920])[:,0]
        self.assertGreater(a[0],a[1])
        x=np.linspace(1900,1940,1001);curves=[(x,(1+.0003*(T-298))*np.exp(-((x-1920)/7)**2)*1e-20) for T in self.temps]
        c=self.model.prepare(curves,self.temps,[1900,1940]);self.assertEqual(c['line_records'],[])
        z=self.model.evaluate(c,[800,800],[1,1.5],[1920]);np.testing.assert_array_equal(z[0],z[1])
    def test_convolution_subtraction_avoids_double_line_area(self):
        # Before nonnegative clipping, each contrast correction integrates to zero
        # on the infinite domain. A very wide numerical interval approaches it.
        from scipy.special import voigt_profile
        x=np.linspace(-300,300,600001);s=.006;g=.03
        contrast=voigt_profile(x,s,g)-voigt_profile(x,np.hypot(s,2/np.sqrt(8*np.log(2))),g)
        self.assertLess(abs(np.trapezoid(contrast,x)),1e-8)
    def test_native_prediction_nonnegative_and_finite(self):
        x=np.linspace(1900,1940,8001);y=self.model.evaluate(self.context,[500,1800],[.7,1.5],x)
        self.assertTrue(np.isfinite(y).all());self.assertTrue((y>=0).all())
    def test_invalid_condition_is_rejected(self):
        for T,P in [(298,1),(800,.5),(800,2.1)]:
            with self.assertRaises(ValueError):self.model.evaluate(self.context,[T],[P],[1920])

if __name__=='__main__':unittest.main()
