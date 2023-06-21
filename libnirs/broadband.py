from __future__ import annotations

from typing import Generator, Iterator, List, NamedTuple, Tuple

import numpy as np
import xarray as xr

from .extinction_coeffs import get_extinction_coeffs
from .statistical import (CentralMoments, StandardMoments,
                          WeightedCentralMoments, weighted_quantile)
from .typed_pmcx import MCX, DetectedPhotons, SaveFlags, VolumetricOutputType
from .utils import jit


class PhotonAnalysisResults(NamedTuple):
    counts: np.ndarray
    phi_time: StandardMoments
    phi: StandardMoments
    opl: StandardMoments
    momentum: StandardMoments
    layer_opl: StandardMoments
    layer_prop: StandardMoments
    layer_momentum: StandardMoments


class PhotonAnalysis(NamedTuple):
    # packet counts
    counts: np.ndarray
    phi_time_distr: CentralMoments
    phi_distr: CentralMoments
    opl_distr: WeightedCentralMoments
    momentum_distr: WeightedCentralMoments
    layer_opl_distr: WeightedCentralMoments
    layer_prop_distr: WeightedCentralMoments
    layer_momentum_distr: WeightedCentralMoments

    @staticmethod
    def alloc(n_detector: int, n_tof: int, n_media: int) -> PhotonAnalysis:
        return PhotonAnalysis(
            counts=np.zeros((n_detector, n_tof), dtype=np.uint64),
            phi_time_distr=CentralMoments.alloc((n_detector, n_tof)),
            phi_distr=CentralMoments.alloc(n_detector),
            opl_distr=WeightedCentralMoments.alloc(n_detector),
            momentum_distr=WeightedCentralMoments.alloc(n_detector),
            layer_opl_distr=WeightedCentralMoments.alloc((n_detector, n_media)),
            layer_prop_distr=WeightedCentralMoments.alloc((n_detector, n_media)),
            layer_momentum_distr=WeightedCentralMoments.alloc((n_detector, n_media)),
        )

    @staticmethod
    def from_arrays(analysis) -> PhotonAnalysis:
        return PhotonAnalysis(
            counts=analysis.counts,
            phi_time_distr=CentralMoments.from_array(analysis.phi_time_distr),
            phi_distr=CentralMoments.from_array(analysis.phi_distr),
            opl_distr=WeightedCentralMoments.from_array(analysis.opl_distr),
            momentum_distr=WeightedCentralMoments.from_array(analysis.momentum_distr),
            layer_opl_distr=WeightedCentralMoments.from_array(analysis.layer_opl_distr),
            layer_prop_distr=WeightedCentralMoments.from_array(analysis.layer_prop_distr),
            layer_momentum_distr=WeightedCentralMoments.from_array(analysis.layer_momentum_distr),
        )

    def to_arrays(self):
        return PhotonAnalysis(
            counts=self.counts,
            phi_time_distr=self.phi_time_distr.to_array(),
            phi_distr=self.phi_distr.to_array(),
            opl_distr=self.opl_distr.to_array(),
            momentum_distr=self.momentum_distr.to_array(),
            layer_opl_distr=self.layer_opl_distr.to_array(),
            layer_prop_distr=self.layer_prop_distr.to_array(),
            layer_momentum_distr=self.layer_momentum_distr.to_array(),
        )

    def finalize(self, nphoton: int) -> PhotonAnalysisResults:
        return PhotonAnalysisResults(
            counts=self.counts.copy(),
            phi_time=self.phi_time_distr.update_to_n(nphoton).standard,
            phi=(self.phi_distr.update_to_n(nphoton).standard),
            opl=(self.opl_distr.standard),
            momentum=(self.momentum_distr.standard),
            layer_opl=(self.layer_opl_distr.standard),
            layer_prop=(self.layer_prop_distr.standard),
            layer_momentum=(self.layer_momentum_distr.standard),
        )


@jit
def analyze_photons(detp, mua, n, c, tof_bin_edges, results: PhotonAnalysis):
    n_tof = len(tof_bin_edges) - 1
    # assert np.isclose(tof_bin_edges[0], 0)
    n_media = len(mua)
    # assert len(mua) == len(n)
    det_bin = detp.detector_id.astype(np.int32) - 1
    layer_opl = n * detp.partial_path.T
    opl = n @ detp.partial_path
    layer_p = layer_opl / opl.reshape((-1, 1))
    # MCX culls photons once they exceed t_end, so the amount detected is negligable
    tof_bin = np.minimum(np.digitize(opl, c * tof_bin_edges), n_tof) - 1
    ln_phi = -mua @ detp.partial_path
    phi = np.exp(ln_phi)
    mom = detp.momentum.sum(axis=0)
    layer_mom = np.ascontiguousarray(detp.momentum.T)
    for i in range(len(det_bin)):
        d_i = det_bin[i]
        t_i = tof_bin[i]
        results.counts[d_i, t_i] += 1
        phi_i = phi[i]
        CentralMoments.push(results.phi_time_distr, phi_i, (d_i, t_i))
        CentralMoments.push(results.phi_distr, phi_i, d_i)
        WeightedCentralMoments.push(results.opl_distr, opl[i], phi_i, d_i)
        WeightedCentralMoments.push(results.momentum_distr, mom[i], phi_i, d_i)
        # WeightedCentralMoments.push(results.layer_opl_distr, layer_opl[i], phi_i, (d_i))
        # WeightedCentralMoments.push(results.layer_prop_distr, layer_p[i], phi_i, (d_i))
        # WeightedCentralMoments.push(results.layer_momentum_distr, layer_mom[i], phi_i, (d_i))
        for j in range(n_media):
            WeightedCentralMoments.push(results.layer_opl_distr, layer_opl[i, j], phi_i, (d_i, j))
            WeightedCentralMoments.push(results.layer_prop_distr, layer_p[i, j], phi_i, (d_i, j))
            WeightedCentralMoments.push(results.layer_momentum_distr, layer_mom[i, j], phi_i, (d_i, j))


class Histogram(NamedTuple):
    histogram: np.ndarray
    bin_edges: np.ndarray


class PhotonsHistograms(NamedTuple):
    opl: Histogram
    mom: Histogram
    layer_opl: Histogram
    layer_mom: Histogram


def photon_histograms(detp, mua, n, n_detector, nbins=128) -> PhotonsHistograms:
    n_media = len(mua)
    # assert len(mua) == len(n)
    det_bin = detp.detector_id.astype(np.int32) - 1
    sidx = np.argsort(det_bin)
    split_idxs = np.add.accumulate(np.bincount(det_bin, minlength=n_detector))[:-1]

    pp = detp.partial_path.T[sidx]
    mm = detp.momentum.T[sidx]
    phi = np.exp(-pp @ mua)
    layer_opl = pp * n
    opl = pp @ n
    layer_mom = mm
    mom = mm.sum(axis=1)

    split_phi = np.split(phi, split_idxs)
    split_opl = np.split(opl, split_idxs)
    split_mom = np.split(mom, split_idxs)
    split_layer_opl = np.split(layer_opl, split_idxs)
    split_layer_mom = np.split(layer_mom, split_idxs)

    it = ((
            np.histogram(o, bins=nbins, weights=p),
            np.histogram(m, bins=nbins, weights=p),
            np.histogramdd(ol, bins=nbins, weights=p),
            np.histogramdd(ml, bins=nbins, weights=p),
        ) for p, o, m, ol, ml in zip(split_phi, split_opl, split_mom, split_layer_opl, split_layer_mom))
    return PhotonsHistograms._make(map(lambda h: Histogram._make(map(np.stack, zip(*h))), zip(*it)))


def photon_quantiles(detp, mua, n, n_detector, nbins=1024):
    n_media = len(mua)
    # assert len(mua) == len(n)
    det_bin = detp.detector_id.astype(np.int32) - 1
    sidx = np.argsort(det_bin)
    split_idxs = np.add.accumulate(np.bincount(det_bin, minlength=n_detector))[:-1]

    pp = detp.partial_path.T[sidx]
    mm = detp.momentum.T[sidx]
    phi = np.exp(-pp @ mua)
    opl = pp @ n
    mom = mm.sum(axis=1)

    split_phi = np.split(phi, split_idxs)
    split_opl = np.split(opl, split_idxs)
    split_mom = np.split(mom, split_idxs)
    
    prepend_zero = lambda a: np.concatenate((np.zeros(1), a))

    q = np.linspace(0, 1, nbins)
    q_phi, q_opl, q_mom = zip(*(
        (np.quantile(w, q), weighted_quantile(o, w, q), weighted_quantile(m, w, q))
        for o, m, w in 
        zip(map(prepend_zero, split_opl), map(prepend_zero, split_mom), map(prepend_zero, split_phi))
    ))

    return q, np.stack(q_phi), np.stack(q_opl), np.stack(q_mom)


@jit(parallel=True)
def analyze_mcx(detp, prop, tof_domain, tau, wavelength, BFi, freq, ntof, nmedia, pcounts, phiTD, phiPhase, g1_top, phiDist, momDist):
    c = 2.998e+11  # speed of light in mm / s
    detBins = detp.detector_id.astype(np.int32) - 1
    layerdist = prop[1:].n * detp.partial_path.T
    totaldist = prop[1:].n @ detp.partial_path
    tofBins = np.minimum(np.digitize(totaldist, c * tof_domain), ntof) - 1
    distBins = np.minimum(np.digitize(layerdist, c * tof_domain), ntof) - 1
    path = -prop[1:].mua @ detp.partial_path
    phis = np.exp(path)
    omega_wavelength = -2 * np.pi * freq / c
    prep = (-2*(2*np.pi*prop[1:].n/(wavelength*1e-6))**2*BFi).astype(np.float32) @ detp.momentum
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


def run_mcx(cfg: MCX, run_count, tof_domain, tau, wavelength, BFi, freq, fslicer):
    seeds = np.random.randint(0xFFFF, size=run_count)
    assert cfg.detpos is not None
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
        detp, fluence, *_ = cfg.run(
            VolumetricOutputType.Fluence,
            SaveFlags.DetectorId | SaveFlags.PartialPath | SaveFlags.Momentum,
        )
        if cfg.unitinmm is not None and cfg.unitinmm != 1:
            detp.partial_path[()] *= cfg.unitinmm  # convert ppath to mm from grid unit
        analyze_mcx(detp, cfg.prop, tof_domain, tau, wavelength, BFi, freq, ntof, nmedia, pcounts, phiTD, phiPhase, g1_top, phiDist, momDist)
        fslice += fluence[fslicer]
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
            "PhiPhase": (["detector"], phiPhase, {"units": "radian", "long_name": "Φ Phase"}),
            "PhiDist": (["detector", "time", "layer"], phiDist, {"long_name": "Φ Distribution"}),
            "g1": (["detector", "tau"], g1),
            "fluence": (["x", "y", "z", "time"], fslice),
            "momDist": (["detector", "time", "layer"], momDist, {"long_name": "Momentum-Transfer Distribution"})
        },
        coords={
            "wavelength": ([], wavelength, {"units": "nanometer", "long_name": "λ"}),
            "time": (["time"], (tof_domain[:-1] + tof_domain[1:]) / 2, {"units": "second"}),
            "tau": (["tau"], tau, {"units": "second", "long_name": "τ"}),
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
