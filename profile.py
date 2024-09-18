from pathlib import Path

import numpy as np
from tqdm.dask import TqdmCallback

import astropy.units as u

from astropy.modeling import models as m
from astropy.nddata import StdDevUncertainty
from sunraster.instr.spice import read_spice_l2_fits
from sospice.calibrate import spice_error

from astropy.modeling.fitting import parallel_fit_dask
from astropy.modeling import fitting

from pyinstrument import Profiler

data_dir = Path("./data")
file = data_dir / "solo_l2_spice_n_ras_20220403t121537_v07_100664022_000.fits"
window = "Ly-gamma-CIII group (Merged)"

spice = read_spice_l2_fits(str(file))[window]

av_cojstant_noise_level, sigmadict = spice_error(data=spice.data, header=spice.meta)
sigma = sigmadict["Total"].value
spice.mask = spice.mask | np.isnan(sigma) | (sigma <= 0)
spice.uncertainty = StdDevUncertainty(sigma)

n = 60
i = 400
j = 0

spice_small = spice[0, :, i : i + n, j : j + n]
spice_small

initial_model = m.Const1D(amplitude=1 * spice.unit) + m.Gaussian1D(
    mean=976.8 * u.AA, amplitude=4 * spice.unit, stddev=1 * u.AA
)

avg_spectra = spice_small.rebin((1, *spice_small.data.shape[1:]))
avg_spectra = avg_spectra[:, 0, 0]  # drop length one dimensions
avg_spectra

weights = 1 / spice_small.uncertainty.array
weights[spice_small.mask] = 0

wave = spice.axis_world_coords("em.wl")[0].to(u.AA)

trf = fitting.TRFLSQFitter()
out = trf(initial_model, wave, spice_small.data[:,0,0]*spice_small.unit, weights=weights[:,0,0], filter_non_finite=True)
print(out)

profile = Profiler()
profile.start()

with TqdmCallback(desc="raster fit"):
    spice_model_fit = parallel_fit_dask(
        model=initial_model,
        fitter=fitting.TRFLSQFitter(),
        fitting_axes=0,
        data=spice_small.data,
        data_unit=spice_small.unit,
        world=(wave,),
        weights=weights,
        fitter_kwargs={"filter_non_finite": True},  # Filter out non-finite values
        scheduler='synchronous'
    )

profile.stop()
profile.write_html('test.html')
