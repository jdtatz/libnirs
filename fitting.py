import numpy as np
import numba as nb
import numba.cuda
from .utils import jit
from .extinction_coeffs import get_extinction_coeffs
from .model import  model_ss, model_fd, model_g2, model_td
from .layered_model import  model_nlayer_ss, model_nlayer_fd, model_nlayer_g2
from .cuda import cuda_model_ss, cuda_model_fd, cuda_model_g2, cuda_model_td, cuda_model_nlayer_ss, cuda_model_nlayer_fd, cuda_model_nlayer_g2
from collections import OrderedDict
import scipy.optimize


@jit
def create_mu(params, extc, waves):
    conc = np.array((params['water_frac'] / 100, params['HbO_conc']*1e-6, params['HbR_conc']*1e-6))
    mua = extc @ conc
    musp = params['scatter_A'] * waves ** -(params['scatter_b'] / 100)
    return mua, musp


@jit
def create_mu_3L(params, extc, waves):
    conc = np.array((
        (params['l1_water_frac'] / 100, params['l2_water_frac'] / 100, params['l3_water_frac'] / 100),
        (params['l1_HbO_conc']*1e-6, params['l2_HbO_conc']*1e-6, params['l3_HbO_conc']*1e-6),
        (params['l1_HbR_conc']*1e-6, params['l2_HbR_conc']*1e-6, params['l3_HbR_conc']*1e-6)
    ))
    mua = extc @ conc
    scA = np.array((params['l1_scatter_A'], params['l2_scatter_A'], params['l3_scatter_A']))
    scb = np.array((params['l1_scatter_b'], params['l2_scatter_b'], params['l3_scatter_b'])) / 100
    musp = scA * waves.reshape((-1, 1)) ** -scb
    return mua, musp


@nb.cuda.jit
def cuda_calc_mu(params, waves, extcs, mua, musp):
    i = nb.cuda.grid(1)
    mua[i] = params.water_frac * extcs[i, 0] + params.HbO_conc * extcs[i, 1] + params.HbR_conc * extcs[i, 2]
    musp[i] = params.scatter_A * waves[i] ** -params.scatter_b


@nb.cuda.jit
def cuda_calc_muN(params, waves, extcs, mua, musp):
    i = nb.cuda.grid(1)
    
    for j in range(params.water_frac.shape[0]):
        mua[i, j] = params.water_frac[j] * extcs[i, 0] + params.HbO_conc[j] * extcs[i, 1] + params.HbR_conc[j] * extcs[i, 2]
        musp[i, j] = params.scatter_A[j] * waves[i] ** -params.scatter_b[j]
    """
    mua[i, 0] = params.l1_water_frac * extcs[i, 0] + params.l1_HbO_conc * extcs[i, 1] + params.l1_HbR_conc * extcs[i, 2]
    mua[i, 1] = params.l2_water_frac * extcs[i, 0] + params.l2_HbO_conc * extcs[i, 1] + params.l2_HbR_conc * extcs[i, 2]
    mua[i, 2] = params.l3_water_frac * extcs[i, 0] + params.l3_HbO_conc * extcs[i, 1] + params.l3_HbR_conc * extcs[i, 2]
    musp[i, 0] = params.l1_scatter_A * waves[i] ** -params.l1_scatter_b
    musp[i, 1] = params.l2_scatter_A * waves[i] ** -params.l2_scatter_b
    musp[i, 2] = params.l3_scatter_A * waves[i] ** -params.l3_scatter_b
    """


base_parameters = {
    'water_frac': {'value': 75, 'min': 50, 'max': 100},
    'HbO_conc': {'value': 60, 'min': 0, 'max': 100},
    'HbR_conc': {'value': 40, 'min': 0, 'max': 100},
    'scatter_A': {'value': 12, 'min': 0, 'max': np.inf},
    'scatter_b': {'value': 35, 'min': 0.1, 'max': 200},
    'ac_fit': {'value': 1, 'min': 0, 'max': 10},
    'BFi': {'value': 6, 'min': 0, 'max': 10},
    'l1_water_frac': {'value': 25, 'min': 0, 'max': 100},
    'l2_water_frac': {'value': 95, 'min': 0, 'max': 100},
    'l3_water_frac': {'value': 75, 'min': 0, 'max': 100},
    'l1_HbO_conc': {'value': 56, 'min': 0, 'max': 100},
    'l2_HbO_conc': {'value': 0.1, 'min': 0, 'max': 100},
    'l3_HbO_conc': {'value': 60, 'min': 0, 'max': 100},
    'l1_HbR_conc': {'value': 18, 'min': 0, 'max': 100},
    'l2_HbR_conc': {'value': 0.1, 'min': 0, 'max': 100},
    'l3_HbR_conc': {'value': 40, 'min': 0, 'max': 100},
    'l1_scatter_A': {'value': 8, 'min': 0, 'max': np.inf},
    'l2_scatter_A': {'value': 0.1, 'min': 0, 'max': np.inf},
    'l3_scatter_A': {'value': 12, 'min': 0, 'max': np.inf},
    'l1_scatter_b': {'value': 35, 'min': 0.1, 'max': 200},
    'l2_scatter_b': {'value': 35, 'min': 0.1, 'max': 200},
    'l3_scatter_b': {'value': 35, 'min': 0.1, 'max': 200},
    'l1_BFi': {'value': 1, 'min': 0, 'max': 20},
    'l2_BFi': {'value': 1e-3, 'min': 0, 'max': 20},
    'l3_BFi': {'value': 6, 'min': 0, 'max': 20},
    'l1': {'value': 3, 'min': 0, 'max': 6},
    'l2': {'value': 4, 'min': 0, 'max': 8},
}


class Fit:
    def __init__(self, rhos, waves, tau, tofs, rdCW, rdTD, ac, phase, g2, n, n_ext, freq, c, beta, tau_0, *, use_cuda=True, cw_slices=None, fd_slices=None, dcs_slices=None, td_slices=None):
        self.rhos = rhos
        self.waves = waves
        self.tau = tau
        self.tofs = tofs
        self.rdCW = rdCW
        self.rdTD = rdTD
        self.ac = ac
        self.phase = phase
        self.g2 = g2
        self.n = n
        self.n_ext = n_ext
        self.freq = freq
        self.c = c
        self.beta = beta
        self.tau_0 = tau_0
        self.use_cuda = use_cuda
        if cw_slices is None:
            cw_slices = slice(None), slice(20, None)
        if fd_slices is None:
            fd_slices = np.logical_or.reduce(np.equal.outer(waves, (671, 692, 706, 727, 759, 784, 812, 830)), axis=1), cw_slices[1]
        if dcs_slices is None:
            dcs_slices = (*fd_slices, slice(None))
        if td_slices is None:
            td_slices = np.logical_or.reduce(np.equal.outer(waves, (756, 897, 975)), axis=1), np.logical_or.reduce(np.equal.outer(rhos, (10,16,24,30)), axis=1), slice(40, None)
        self.cw_slices = cw_slices
        self.fd_slices = fd_slices
        self.dcs_slices = dcs_slices
        self.td_slices = td_slices

    def mesh(self, slices, *args):
        f = (lambda v: (v, nb.cuda.to_device(v))) if self.use_cuda else (lambda v: v)
        yield from (f(v.flatten()) for v in np.meshgrid(*(arg[s] for s, arg in zip(slices, args)), indexing='ij'))
        
    def fix_params(self, params, dtype):
        params = params.view(dtype)[0].copy()
        params['water_frac'] /= 100
        params['HbO_conc'] *= 1e-6
        params['HbR_conc'] *= 1e-6
        # params['scatter_A'] *= 1
        params['scatter_b'] /= 100
        if 'BFi' in dtype.names:
            params['BFi'] *= 1e-6
        if 'depths' in dtype.names:
            params['depths'] = np.add.accumulate(params['depths'])
        return params
        
    def fit(self, dtype, init, bounds, residual):
        param_names = list(dtype.names)
        opt = scipy.optimize.least_squares(residual, init.view(np.float64), bounds=bounds.view(np.float64).reshape((2, -1)), verbose=0, max_nfev=None)
        r = opt.x.view(dtype)[0]
        return OrderedDict([(k, r[k]) for k in dtype.names]) 
    
    def create_fd_residual(self):
        dtype = np.dtype([
            ('water_frac', 'f8'),
            ('HbO_conc', 'f8'),
            ('HbR_conc', 'f8'),
            ('scatter_A', 'f8'),
            ('scatter_b', 'f8'),
            ('ac_fit', 'f8'),
        ])
        init = np.array([(75, 60, 40, 12, 35, 1)], dtype=dtype)
        bounds = np.array([(0, 0, 0, 0, 0.1, 0), (100, 100, 100, np.inf, 200, 10)], dtype=dtype)
        if self.use_cuda:
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            ydata = sigma = np.concatenate((self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            d_fd_musp = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            fd_grid = 1 + d_fd.size // 64, 64
            def residual(params):
                params = self.fix_params(params, dtype)
                cuda_calc_mu[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                cuda_model_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, n, n_ext, freq, c, d_fd)
                fd = d_fd.copy_to_host()
                fit = np.concatenate((np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        else:
            fd_waves, fd_rhos = self.mesh(self.fd_slices, self.waves, self.rhos)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=0)
            ydata = sgima = np.concatenate((self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            def residual(params):
                params = self.fix_params(params, dtype)
                fd_mua, fd_musp = create_mu(params, fd_extc, fd_waves)
                fd = model_fd(fd_rhos, fd_mua, fd_musp, n, n_ext, freq, c)
                fit = np.concatenate((np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        return dtype, init, bounds, residual

    def create_bb_residual(self):
        dtype = np.dtype([
            ('water_frac', 'f8'),
            ('HbO_conc', 'f8'),
            ('HbR_conc', 'f8'),
            ('scatter_A', 'f8'),
            ('scatter_b', 'f8'),
            ('ac_fit', 'f8'),
        ])
        init = np.array([(75, 60, 40, 12, 35, 1)], dtype=dtype)
        bounds = np.array([(0, 0, 0, 0, 0.1, 0), (100, 100, 100, np.inf, 200, 10)], dtype=dtype)
        if self.use_cuda:
            (cw_waves, d_cw_waves), (cw_rhos, d_cw_rhos) = self.mesh(self.cw_slices, self.waves, self.rhos)
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_cw_extc = nb.cuda.to_device(cw_extc)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            ydata = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            sigma = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_ss = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_mua = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_musp = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            d_fd_musp = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            ss_grid = 1 + d_ss.size // 64, 64
            fd_grid = 1 + d_fd.size // 64, 64
            def residual(params):
                params = self.fix_params(params, dtype)
                cuda_calc_mu[ss_grid](params, d_cw_waves, d_cw_extc, d_cw_mua, d_cw_musp)
                cuda_calc_mu[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                cuda_model_ss[ss_grid](d_cw_rhos, d_cw_mua, d_cw_musp, n, n_ext, d_ss)
                cuda_model_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, n, n_ext, freq, c, d_fd)
                ss = d_ss.copy_to_host()
                fd = d_fd.copy_to_host()
                fit = np.concatenate((ss, np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        else:
            pass
        return dtype, init, bounds, residual
    
    def create_bb_2l_residual(self):
        dtype = np.dtype([
            ('water_frac', 'f8', 2),
            ('HbO_conc', 'f8', 2),
            ('HbR_conc', 'f8', 2),
            ('scatter_A', 'f8', 2),
            ('scatter_b', 'f8', 2),
            ('depths', 'f8', (1,)),
            ('ac_fit', 'f8'),
        ])
        init = np.array([((25, 75), (56, 60), (18, 40), (8, 12), (35, 35), (5,), 1)], dtype=dtype)
        bounds = np.array([((0, 0), (0, 0), (0, 0), (0, 0), (0.1, 0.1), (0,), 0), 
                           ((100, 100), (100, 100), (100, 100), (np.inf, np.inf), (200, 200), (10,), 10)], dtype=dtype)
        if self.use_cuda:
            (cw_waves, d_cw_waves), (cw_rhos, d_cw_rhos) = self.mesh(self.cw_slices, self.waves, self.rhos)
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_cw_extc = nb.cuda.to_device(cw_extc)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            ydata = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            sigma = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_ss = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_mua = nb.cuda.device_array((cw_rhos.size, 2), dtype=np.float64)
            d_cw_musp = nb.cuda.device_array((cw_rhos.size, 2), dtype=np.float64)
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array((fd_rhos.size, 2), dtype=np.float64)
            d_fd_musp = nb.cuda.device_array((fd_rhos.size, 2), dtype=np.float64)
            ss_grid = 1 + d_ss.size // 64, 64
            fd_grid = 1 + d_fd.size // 64, 64
            def residual(params):
                params = self.fix_params(params, dtype)
                cuda_calc_muN[ss_grid](params, d_cw_waves, d_cw_extc, d_cw_mua, d_cw_musp)
                cuda_calc_muN[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                cuda_model_nlayer_ss[ss_grid](d_cw_rhos, d_cw_mua, d_cw_musp, params['depths'], n, n_ext, 20, 20, 1e-16, d_ss)
                cuda_model_nlayer_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, params['depths'], freq, c, n, n_ext, 20, 20, 1e-16, d_fd)
                ss = d_ss.copy_to_host()
                fd = d_fd.copy_to_host()
                fit = np.concatenate((ss, np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        else:
            pass
        return dtype, init, bounds, residual

    def create_bb_3l_residual(self):
        dtype = np.dtype([
            ('water_frac', 'f8', 3),
            ('HbO_conc', 'f8', 3),
            ('HbR_conc', 'f8', 3),
            ('scatter_A', 'f8', 3),
            ('scatter_b', 'f8', 3),
            ('depths', 'f8', 2),
            ('ac_fit', 'f8'),
        ])
        init = np.array([((25, 95, 75), (56, 0.1, 60), (18, 0.1, 40), (8, 0.1, 12), (35, 35, 35), (3, 4), 1)], dtype=dtype)
        bounds = np.array([((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0.1, 0.1, 0.1), (0, 0), 0), 
                           ((100, 100, 100), (100, 100, 100), (100, 100, 100), (np.inf, np.inf, np.inf), (200, 200, 200), (6, 8), 10)], dtype=dtype)

        if self.use_cuda:
            (cw_waves, d_cw_waves), (cw_rhos, d_cw_rhos) = self.mesh(self.cw_slices, self.waves, self.rhos)
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_cw_extc = nb.cuda.to_device(cw_extc)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            ydata = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            sigma = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_ss = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_mua = nb.cuda.device_array((cw_rhos.size, 3), dtype=np.float64)
            d_cw_musp = nb.cuda.device_array((cw_rhos.size, 3), dtype=np.float64)
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array((fd_rhos.size, 3), dtype=np.float64)
            d_fd_musp = nb.cuda.device_array((fd_rhos.size, 3), dtype=np.float64)
            ss_grid = 1 + d_ss.size // 64, 64
            fd_grid = 1 + d_fd.size // 64, 64
            def residual(params):
                params = self.fix_params(params, dtype)
                cuda_calc_muN[ss_grid](params, d_cw_waves, d_cw_extc, d_cw_mua, d_cw_musp)
                cuda_calc_muN[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                cuda_model_nlayer_ss[ss_grid](d_cw_rhos, d_cw_mua, d_cw_musp, params['depths'], n, n_ext, 20, 20, 1e-16, d_ss)
                cuda_model_nlayer_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, params['depths'], freq, c, n, n_ext, 20, 20, 1e-16, d_fd)
                ss = d_ss.copy_to_host()
                fd = d_fd.copy_to_host()
                fit = np.concatenate((ss, np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        else:
            pass
        return dtype, init, bounds, residual

    def create_bb_dcs_3l_residual(self):
        dtype = np.dtype([
            ('water_frac', 'f8', 3),
            ('HbO_conc', 'f8', 3),
            ('HbR_conc', 'f8', 3),
            ('scatter_A', 'f8', 3),
            ('scatter_b', 'f8', 3),
            ('BFi', 'f8', 3),
            ('depths', 'f8', 2),
            ('ac_fit', 'f8'),
        ])
        init = np.array([((25, 95, 75), (56, 0.1, 60), (18, 0.1, 40), (8, 0.1, 12), (35, 35, 35), (1, 1e-3, 6), (3, 4), 1)], dtype=dtype)
        bounds = np.array([((0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0.1, 0.1, 0.1), (0, 0, 0), (0, 0), 0), 
                           ((100, 100, 100), (100, 100, 100), (100, 100, 100), (np.inf, np.inf, np.inf), (200, 200, 200), (20, 20, 20), (6, 8), 10)], dtype=dtype)
        if self.use_cuda:
            (cw_waves, d_cw_waves), (cw_rhos, d_cw_rhos) = self.mesh(self.cw_slices, self.waves, self.rhos)
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            (dcs_waves, d_dcs_waves), (dcs_rhos, d_dcs_rhos), (dcs_tau, d_dcs_tau) = self.mesh(self.dcs_slices, self.waves, self.rhos, self.tau)
            cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            dcs_extc = get_extinction_coeffs(dcs_waves, 'water', 'HbO', 'HbR', axis=1)
            d_cw_extc = nb.cuda.to_device(cw_extc)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            d_dcs_extc = nb.cuda.to_device(dcs_extc)
            ydata = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat, self.g2[self.dcs_slices].flat))
            sigma = np.concatenate((self.rdCW[self.cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10, self.g2[self.dcs_slices].flat))
            n, n_ext, freq, c, beta, tau_0 = self.n, self.n_ext, self.freq, self.c, self.beta, self.tau_0
            d_ss = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_mua = nb.cuda.device_array((cw_rhos.size, 3), dtype=np.float64)
            d_cw_musp = nb.cuda.device_array((cw_rhos.size, 3), dtype=np.float64)
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array((fd_rhos.size, 3), dtype=np.float64)
            d_fd_musp = nb.cuda.device_array((fd_rhos.size, 3), dtype=np.float64)
            d_dcs = nb.cuda.device_array(dcs_rhos.size, dtype=np.float64)
            d_dcs_mua = nb.cuda.device_array((dcs_rhos.size, 3), dtype=np.float64)
            d_dcs_musp = nb.cuda.device_array((dcs_rhos.size, 3), dtype=np.float64)
            ss_grid = 1 + d_ss.size // 64, 64
            fd_grid = 1 + d_fd.size // 64, 64
            dcs_grid = 1 + d_dcs.size // 64, 64
            d_dcs_waves2 = nb.cuda.to_device(dcs_waves * 1e-6)
            def residual(params):
                params = self.fix_params(params, dtype)
                cuda_calc_muN[ss_grid](params, d_cw_waves, d_cw_extc, d_cw_mua, d_cw_musp)
                cuda_calc_muN[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                cuda_calc_muN[dcs_grid](params, d_dcs_waves, d_dcs_extc, d_dcs_mua, d_dcs_musp)
                cuda_model_nlayer_ss[ss_grid](d_cw_rhos, d_cw_mua, d_cw_musp, params['depths'], n, n_ext, 20, 20, 1e-16, d_ss)
                cuda_model_nlayer_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, params['depths'], freq, c, n, n_ext, 20, 20, 1e-16, d_fd)
                cuda_model_nlayer_g2[dcs_grid](d_dcs_rhos, d_dcs_tau, d_dcs_mua, d_dcs_musp, params['depths'], params['BFi'], d_dcs_waves2, n, n_ext, beta, tau_0, 20, 20, 1e-16, d_dcs)
                ss = d_ss.copy_to_host()
                fd = d_fd.copy_to_host()
                dcs = d_dcs.copy_to_host()
                fit = np.concatenate((ss, np.abs(fd) * params['ac_fit'], np.angle(fd), dcs))
                return (fit - ydata) / sigma
        else:
            pass
        return dtype, init, bounds, residual
