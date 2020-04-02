import numpy as np
from pymcx import MCX
from multiprocessing import Process, Queue, Manager
from .utils import jit


@jit(parallel=True)
def analyze_mcx(detp, prop, tof_domain, tau, wavelength, BFi, freq, ndet, ntof, nmedia, pcounts, paths, phiTD, phiFD, g1_top, phiDist):
    c = 2.998e+11  # speed of light in mm / s
    detBins = detp[0].astype(np.int32) - 1
    layerdist = prop[1:, 3] * detp[1:(1+nmedia)].T
    totaldist = prop[1:, 3] @ detp[1:(1+nmedia)]
    tofBins = np.minimum(np.digitize(totaldist, c * tof_domain), ntof) - 1
    distBins = np.minimum(np.digitize(layerdist, c * tof_domain), ntof) - 1
    path = -prop[1:, 0] @ detp[1:(1+nmedia)]
    phis = np.exp(path)
    fds = np.exp((-prop[1:, 0] - 2j * np.pi * freq * prop[1:, 3] / c).astype(np.complex64) @ detp[1:(1+nmedia)].astype(np.complex64))
    prep = (-2*(2*np.pi*prop[1:, 3]/(wavelength*1e-6))**2*BFi).astype(np.float32) @ detp[(1+nmedia):(1+2*nmedia)]
    big = np.exp(prep * tau.reshape((len(tau), 1)) + path)
    for i in range(len(detBins)):
        pcounts[detBins[i], tofBins[i]] += 1
        paths[detBins[i], tofBins[i]] += detp[1:(1+nmedia), i]
        phiFD[detBins[i]] += fds[i]
        phiTD[detBins[i], tofBins[i]] += phis[i]
        for l in range(nmedia):
            phiDist[detBins[i], distBins[i, l], l] += phis[i] * layerdist[i, l] / totaldist[i]
        g1_top[detBins[i]] += big[:, i]


def run_mcx(cfg, run_count, tof_domain, tau, wavelength, BFi, freq, fslicer):
    seeds = np.random.randint(0xFFFF, size=run_count)
    ndet, ntof, nmedia = len(cfg.detpos), len(tof_domain) - 1, len(cfg.prop) - 1
    phiTD = np.zeros((ndet, ntof), np.float64)
    phiFD = np.zeros(ndet, np.complex128)
    paths = np.zeros((ndet, ntof, nmedia), np.float64)
    pcounts = np.zeros((ndet, ntof), np.int64)
    g1_top = np.zeros((ndet, len(tau)), np.float64)
    phiDist = np.zeros((ndet, ntof, nmedia), np.float64)
    fslice = 0
    for seed in seeds:
        cfg.seed = seed
        result = cfg.run(2, True)
        print(result['stdout'])
        detp = result["detphoton"]
        if detp.shape[1] >= cfg.maxdetphoton:
            raise Exception("Too many photons detected: {}".format(detp.shape[1]))
        if cfg.unitinmm != 1:
            detp[2:(2+nmedia)] *= cfg.unitinmm  # ppath to mm from grid unit
        analyze_mcx(detp, cfg.prop, tof_domain, tau, wavelength, BFi, freq, ndet, ntof, nmedia, pcounts, paths, phiTD, phiFD, g1_top, phiDist)
        fslice += result["fluence"][fslicer]
        del detp
        del result
    fslice /= run_count
    paths /= pcounts[:, :, np.newaxis]
    g1 = g1_top / np.sum(phiTD, axis=1)[:, np.newaxis]
    phiDist /= np.sum(phiTD, axis=1)[:, np.newaxis, np.newaxis]
    return {'wavelength': wavelength, 'Photons': pcounts, 'Paths': paths, 'PhiTD': phiTD, 'PhiFD': phiFD, 'PhiDist': phiDist, 'Seeds': seeds, 'Slice': fslice, 'g1': g1}


def run_broadband_thread(filename, init_cfg, wavelength_queue, create_prop, gpuid, results, run_count, tof_domain, tau, BFi, freq, fslicer):
    cfg = MCX(**init_cfg._config, gpuid=gpuid)
    print('start', filename, gpuid)
    while True:
        wavelength = wavelength_queue.get()
        if wavelength is None:
            break
        print(filename, wavelength, gpuid)
        cfg.prop = create_prop(wavelength)
        results[wavelength] = run_many(cfg, run_count, tof_domain, tau, wavelength, BFi, freq, fslicer)


def run_threaded_broadband(gpu_count, filename, init_cfg, waves, create_prop, run_count, tof_domain, tau, BFi, freq, fslicer):
    with Manager() as manager:
        results = manager.dict()
        wavelength_queue = Queue()
        procs = []
        for w in waves:
            wavelength_queue.put(w)
        for i in range(1, 1 + gpu_count):
            wavelength_queue.put(None)
            procs.append(Process(target=run_broadband_thread, args=(filename, init_cfg, wavelength_queue, create_prop, i, results, run_count, tof_domain, tau, BFi, freq, fslicer)))
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        results = {**results}
    print('done')
    keys = 'wavelength', 'Photons', 'Paths', 'PhiTD', 'PhiFD', 'PhiDist', 'Seeds', 'Slice', 'g1'
    ws = sorted(list(results.keys()))
    np.savez_compressed(filename, init_cfg=init_cfg, **{key: np.stack([results[w][key] for w in ws]) for key in keys})


def create_props(layers, lprops, wavelen):
    media = np.empty((1+len(layers), 4), np.float32)
    media[0] = 0, 0, 1, 1
    for i, l in enumerate(layers):
        lp = lprops[l]
        g = lp['g']
        mua = sum(get_extinction_coeffs(wavelen, k) * lp['components'][k] for k in ext_coeff)
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
