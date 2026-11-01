# Output resolution

The ExtraTrees envelope `B` is fitted to reference cross-sections smoothed with a Gaussian full width at half maximum (FWHM) of 2 cm⁻¹. Applying another 2 cm⁻¹ convolution to `B` broadens that learned response again and reduces the heights of narrow peaks. Band comparisons therefore request the 2 cm⁻¹ response directly.

Both `evaluate` and `predict` accept `output_fwhm_cm1`, a nonnegative Gaussian FWHM in cm⁻¹. Its default is zero, which preserves the existing fine-grid predictions and single-wavenumber values. At a requested FWHM `r` between zero and 2 cm⁻¹, the prediction is

`max(B + sum(G_r * L - G_2 * L), 0)`.

Here `L` is the predicted intrinsic profile of an eligible line observed in all three cold input spectra; `G_r * L` is that profile convolved with a Gaussian of FWHM `r`; and `G_0 * L = L`. Each Voigt convolution is evaluated analytically by adding Gaussian variances. At `r = 2`, the line contrasts cancel and the output is exactly the learned envelope. At `r > 2`, the envelope is convolved only with the additional Gaussian width `sqrt(r² - 2²)`.

This is a resolution-aware approximation. The unresolved background retains the envelope's 2 cm⁻¹ response at finer output settings; only eligible cold-observed line contrasts gain finer detail. Consequently, outputs at different requested widths must not be described as exact convolutions of one completely resolved intrinsic spectrum. The input spectra do not determine every unobserved hot transition.

The resolution correction changes no fitted parameter, training species, test species, cold-input condition, or line-selection threshold. The default fine-grid and probe values are unchanged. Held-out reference spectra are used only after prediction to calculate errors.

```python
# Same three-spectrum context and fitted weights, with the response used for
# the band comparison requested explicitly:
band = model.evaluate(context, temperatures, pressures, wavenumbers,
                      output_fwhm_cm1=2.0)
```
