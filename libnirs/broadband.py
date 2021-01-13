import numpy as np
import xarray as xr
from pint import get_application_registry
from pymcx import MCX, SaveFlags, SrcType, DetectedPhotons
from .utils import jit
from .extinction_coeffs import get_extinction_coeffs


@jit(parallel=True)
def analyze_mcx(detp, prop, tof_domain, tau, wavelength, BFi, freq, ntof, nmedia, pcounts, phiTD, phiPhase, g1_top, phiDist, momDist):
    c = 2.998e+11  # speed of light in mm / s
    detBins = detp.detector_id.astype(np.int32) - 1
    layerdist = prop[1:, 3] * detp.partial_path.T
    totaldist = prop[1:, 3] @ detp.partial_path
    tofBins = np.minimum(np.digitize(totaldist, c * tof_domain), ntof) - 1
    distBins = np.minimum(np.digitize(layerdist, c * tof_domain), ntof) - 1
    path = -prop[1:, 0] @ detp.partial_path
    phis = np.exp(path)
    omega_wavelength = -2 * np.pi * freq / c
    prep = (-2*(2*np.pi*prop[1:, 3]/(wavelength*1e-6))**2*BFi).astype(np.float32) @ detp.momentum
    big = np.exp(prep * tau.reshape((len(tau), 1)) + path)
    mom_prep = phis.reshape((len(phis), 1)) * detp.momentum.T
    for i in range(len(detBins)):
        pcounts[detBins[i], tofBins[i]] += 1
        phiPhase[detBins[i]] += phis[i] * omega_wavelength * totaldist[i]
        phiTD[detBins[i], tofBins[i]] += phis[i]
        for l in range(nmedia):
            phiDist[detBins[i], distBins[i, l], l] += phis[i] * layerdist[i, l] / totaldist[i]
        g1_top[detBins[i]] += big[:, i]
        momDist[detBins[i], tofBins[i]] += mom_prep[i]


def run_mcx(cfg, run_count, tof_domain, tau, wavelength, BFi, freq, fslicer, ureg=None):
    if ureg is None:
        ureg = get_application_registry()
    seeds = np.random.randint(0xFFFF, size=run_count)
    ndet, ntof, nmedia = len(cfg.detpos), len(tof_domain) - 1, len(cfg.prop) - 1
    phiTD = np.zeros((ndet, ntof), np.float64)
    phiPhase = np.zeros(ndet, np.float64)
    pcounts = np.zeros((ndet, ntof), np.int64)
    g1_top = np.zeros((ndet, len(tau)), np.float64)
    phiDist = np.zeros((ndet, ntof, nmedia), np.float64)
    momDist = np.zeros((ndet, ntof, nmedia), np.float64)
    fslice = 0
    for seed in seeds:
        cfg.seed = seed
        cfg.savedetflag = SaveFlags.DetectorId | SaveFlags.PartialPath | SaveFlags.Momentum
        cfg.issave2pt = True
        cfg.issavedet = True
        cfg.run()
        print(cfg.stdout)
        detp = cfg.detphoton
        if cfg.unitinmm != 1:
            detp.partial_path[()] *= cfg.unitinmm  # convert ppath to mm from grid unit
        analyze_mcx(detp, cfg.prop, tof_domain, tau, wavelength, BFi, freq, ntof, nmedia, pcounts, phiTD, phiPhase, g1_top, phiDist, momDist)
        fslice += cfg.fluence[fslicer]
    nphoton = run_count * cfg.nphoton
    fslice /= run_count
    g1 = g1_top / np.sum(phiTD, axis=1)[:, np.newaxis]
    phiDist /= np.sum(phiTD, axis=1)[:, np.newaxis, np.newaxis]
    phiPhase /= np.sum(phiTD, axis=1)
    momDist /= np.sum(phiTD, axis=1)[:, np.newaxis, np.newaxis]
    return xr.Dataset(
        {
            "seeds": (["runs"], seeds),
            "Photons": (["detector", "time"], pcounts),
            "PhiTD": (["detector", "time"], phiTD / nphoton, {"long_name": "Φ"}),
            "PhiPhase": (["detector"], phiPhase, {"units": ureg.rad, "long_name": "Φ Phase"}),
            "PhiDist": (["detector", "time", "layer"], phiDist, {"long_name": "Φ Distribution"}),
            "g1": (["detector", "tau"], g1),
            "fluence": (["x", "y", "z", "time"], fslice),
            "momDist": (["detector", "time", "layer"], momDist, {"long_name": "Momentum-Transfer Distribution"})
        },
        coords={
            "wavelength": ([], wavelength, {"units": ureg.nanometer, "long_name": "λ"}),
            "time": (["time"], (tof_domain[:-1] + tof_domain[1:]) / 2, {"units": ureg.second}),
            "tau": (["tau"], tau, {"units": ureg.second, "long_name": "τ"}),
        }
    )


def create_props(layers, lprops, wavelen):
    media = np.empty((1+len(layers), 4), np.float32)
    media[0] = 0, 0, 1, 1
    for i, l in enumerate(layers):
        lp = lprops[l]
        g = lp['g']
        mua = sum(get_extinction_coeffs(wavelen, k) * v for k, v in lp['components'].items())
        mus = lp['Scatter A'] * wavelen ** -lp['Scatter b'] / (1 - g)
        media[1+i] = mua, mus, g, lp['n']
    return media


def _create_conc(water, HbT, stO2):
    return {'water': water, 'HbO': HbT * stO2, 'HbR': HbT * (1 - stO2)}


base_tissue_properties = {
    "Brain": {"components": _create_conc(water=0.75, HbT=103e-6, stO2=0.65), "Scatter A": 12.317063907, "Scatter b": 0.35, "BFi": 6e-6, "g": 0.9, "n": 1.4},
    "CSF": {"components": _create_conc(water=1, HbT=0, stO2=0), "Scatter A": 0.098536511, "Scatter b": 0.35, "BFi": 1e-9, "g": 0.9, "n": 1.4},
    "Scalp/Skull": {"components": _create_conc(water=0.26, HbT=75e-6, stO2=0.75), "Scatter A": 8.868286013, "Scatter b": 0.35, "BFi": 1e-6, "g": 0.9, "n": 1.4},
    "Breast": {"components": _create_conc(water=0.51, HbT=30e-6, stO2=0.70), "Scatter A": 2700., "Scatter b": 1.2, "BFi": 6e-6, "g": 0.9, "n": 1.4},
}

base_mcx_cfg = {
    'autopilot': True,
    'gpuid': 0,
    'nphoton': 3e8,
    'maxdetphoton': 1.5e8,
    'tstart': 0,
    'tend': 5e-9,
    'tstep': 1e-10,
    'issrcfrom0': True,
    'isreflect': True,
    'ismomentum': True,
    'issaveexit': False,
}
