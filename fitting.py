import numpy as np
import numba as nb
import numba.cuda
from .utils import jit
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
    """
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
        
        self.cw_waves, self.cw_rhos = self.mesh(cw_slices, waves, rhos)
        self.fd_waves, self.fd_rhos = self.mesh(fd_slices, waves, rhos)
        self.dcs_waves, self.dcs_rhos, self.dcs_tau = self.mesh(dcs_slices, waves, rhos)
        self.td_waves, self.td_rhos, self.td_tofs = self.mesh(td_slices, waves, rhos)
        
        self.cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
        self.fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
        self.dcs_extc = get_extinction_coeffs(dcs_waves, 'water', 'HbO', 'HbR', axis=1)
        self.td_extc = get_extinction_coeffs(dcs_waves, 'water', 'HbO', 'HbR', axis=1)
        
        if use_cuda:
            self.d_cw_waves, self.d_cw_rhos = nb.cuda.to_device(self.cw_waves), nb.cuda.to_device(self.cw_rhos)
            self.d_fd_waves, self.d_fd_rhos = nb.cuda.to_device(self.fd_waves), nb.cuda.to_device(self.fd_rhos)
            self.d_dcs_waves, self.d_dcs_rhos, self.d_dcs_tau = nb.cuda.to_device(self.dcs_waves), nb.cuda.to_device(self.dcs_rhos), nb.cuda.to_device(self.dcs_tau)
            self.d_td_waves, self.d_td_rhos, self.d_td_tofs = nb.cuda.to_device(self.td_waves), nb.cuda.to_device(self.td_rhos), nb.cuda.to_device(self.td_tofs)
            self.d_cw_extc = nb.cuda.to_device(self.cw_extc)
            self.d_fd_extc = nb.cuda.to_device(self.fd_extc)
            self.d_dcs_extc = nb.cuda.to_device(self.dcs_extc)
            self.d_td_extc = nb.cuda.to_device(self.td_extc)
    
    def mesh(self, slices, *args):
        f = (lambda v: (v, nb.cuda.to_device(v))) if self.use_cuda else (lambda v: v)
        yield from (f(v.flatten()) for v in np.meshgrid(*(arg[s] for s, arg in zip(slices, args)), indexing='ij'))
        
    def fit(self, create_residual_func, data):
        dtype, residual = create_residual_func(data)
        param_names = list(dtype.names)
        params = [base_parameters[k] for k in param_names]
        init = np.array([p['value'] for p in params], dtype=np.float64)
        bounds = [p['min'] for p in params], [p['max'] for p in params]
        opt = scipy.optimize.least_squares(residual, init, bounds=bounds, verbose=0, max_nfev=None)
        return OrderedDict(zip(param_names, opt.x))
    
    def create_fd_residual(self):
        fields = 'water_frac', 'HbO_conc', 'HbR_conc', 'scatter_A', 'scatter_b', 'ac_fit'
        dtype = np.dtype([(k, 'f8') for k in fields])
        if self.use_cuda:
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_extc = nb.cuda.to_device(fd_extc)
            ydata = sgima = np.concatenate((self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_mua = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            d_musp = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            grid = 1 + d_fd.size // 64, 64
            def residual(oparams):
                params = oparams.view(dtype)[0].copy()
                params['water_frac'] /= 100
                params['HbO_conc'] *= 1e-6
                params['HbR_conc'] *= 1e-6
                params['scatter_b'] /= 100
                cuda_calc_mu[grid](params, d_waves, d_extc, d_mua, d_musp)
                cuda_model_fd[grid](d_rhos, d_mua, d_musp, n, n_ext, freq, c, d_fd)
                fd = d_fd.copy_to_host()
                fit = np.concatenate((np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        else:
            fd_waves, fd_rhos = self.mesh(self.fd_slices, self.waves, self.rhos)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=0)
            ydata = sgima = np.concatenate((self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            @jit
            def residual(oparams):
                params = oparams.view(dtype)[0].copy()
                params['water_frac'] /= 100
                params['HbO_conc'] *= 1e-6
                params['HbR_conc'] *= 1e-6
                params['scatter_b'] /= 100
                fd_mua, fd_musp = create_mu(params, fd_extc, fd_waves)
                fd = model_fd(fd_rhos, fd_mua, fd_musp, n, n_ext, freq, c)
                fit = np.concatenate((np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        return dtype, residual

    def create_bb_residual(self):
        fields = 'water_frac', 'HbO_conc', 'HbR_conc', 'scatter_A', 'scatter_b', 'ac_fit'
        dtype = np.dtype([(k, 'f8') for k in fields])
        if self.use_cuda:
            (cw_waves, d_cw_waves), (cw_rhos, d_cw_rhos) = self.mesh(self.cw_slices, self.waves, self.rhos)
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_cw_extc = nb.cuda.to_device(cw_extc)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            ydata = np.concatenate((self.rdCW[cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            sigma = np.concatenate((self.rdCW[cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_ss = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_mua = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_musp = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            d_fd_musp = nb.cuda.device_array(fd_rhos.size, dtype=np.float64)
            ss_grid = 1 + d_ss.size // 64, 64
            fd_grid = 1 + d_fd.size // 64, 64
            def residual(oparams):
                params = oparams.view(dtype)[0].copy()
                params['water_frac'] /= 100
                params['HbO_conc'] *= 1e-6
                params['HbR_conc'] *= 1e-6
                params['scatter_b'] /= 100
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
        return dtype, residual
    
    def create_bb_3l_residual(self):
        fields = (
            'ac_fit',
            'l1',
            'l1_HbO_conc',
            'l1_HbR_conc',
            'l1_scatter_A',
            'l1_scatter_b',
            'l1_water_frac',
            'l2',
            'l2_HbO_conc',
            'l2_HbR_conc',
            'l2_scatter_A',
            'l2_scatter_b',
            'l2_water_frac',
            'l3_HbO_conc',
            'l3_HbR_conc',
            'l3_scatter_A',
            'l3_scatter_b',
            'l3_water_frac',
        )
        dtype = np.dtype([(k, 'f8') for k in fields])
        if self.use_cuda:
            (cw_waves, d_cw_waves), (cw_rhos, d_cw_rhos) = self.mesh(self.cw_slices, self.waves, self.rhos)
            (fd_waves, d_fd_waves), (fd_rhos, d_fd_rhos) = self.mesh(self.fd_slices, self.waves, self.rhos)
            cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR', axis=1)
            fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR', axis=1)
            d_cw_extc = nb.cuda.to_device(cw_extc)
            d_fd_extc = nb.cuda.to_device(fd_extc)
            ydata = np.concatenate((self.rdCW[cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat))
            sigma = np.concatenate((self.rdCW[cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10))
            n, n_ext, freq, c = self.n, self.n_ext, self.freq, self.c
            d_ss = nb.cuda.device_array(cw_rhos.size, dtype=np.float64)
            d_cw_mua = nb.cuda.device_array((cw_rhos.size, 3), dtype=np.float64)
            d_cw_musp = nb.cuda.device_array((cw_rhos.size, 3), dtype=np.float64)
            d_fd = nb.cuda.device_array(fd_rhos.size, dtype=np.complex128)
            d_fd_mua = nb.cuda.device_array((fd_rhos.size, 3), dtype=np.float64)
            d_fd_musp = nb.cuda.device_array((fd_rhos.size, 3), dtype=np.float64)
            ss_grid = 1 + d_ss.size // 64, 64
            fd_grid = 1 + d_fd.size // 64, 64
            def residual(oparams):
                params = oparams.view(dtype)[0].copy()
                params['l1_water_frac'] /= 100
                params['l2_water_frac'] /= 100
                params['l3_water_frac'] /= 100
                params['l1_HbO_conc'] *= 1e-6
                params['l2_HbO_conc'] *= 1e-6
                params['l3_HbO_conc'] *= 1e-6
                params['l1_HbR_conc'] *= 1e-6
                params['l2_HbR_conc'] *= 1e-6
                params['l3_HbR_conc'] *= 1e-6
                params['l1_scatter_b'] /= 100
                params['l2_scatter_b'] /= 100
                params['l3_scatter_b'] /= 100
                cuda_calc_muN[ss_grid](params, d_cw_waves, d_cw_extc, d_cw_mua, d_cw_musp)
                cuda_calc_muN[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                depths = np.array((params['l1'], params['l1'] + params['l2']))
                cuda_model_nlayer_ss[ss_grid](d_cw_rhos, d_cw_mua, d_cw_musp, depths, n, n_ext, 20, 20, 1e-16, d_ss)
                cuda_model_nlayer_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, depths, freq, c, n, n_ext, 20, 20, 1e-16, d_fd)
                ss = d_ss.copy_to_host()
                fd = d_fd.copy_to_host()
                fit = np.concatenate((ss, np.abs(fd) * params['ac_fit'], np.angle(fd)))
                return (fit - ydata) / sigma
        else:
            pass
        return dtype, residual

    def create_bb_3l_residual(self):
        fields = (
            'ac_fit',
            'l1',
            'l1_HbO_conc',
            'l1_HbR_conc',
            'l1_scatter_A',
            'l1_scatter_b',
            'l1_water_frac',
            'l1_BFi',
            'l2',
            'l2_HbO_conc',
            'l2_HbR_conc',
            'l2_scatter_A',
            'l2_scatter_b',
            'l2_water_frac',
            'l2_BFi',
            'l3_HbO_conc',
            'l3_HbR_conc',
            'l3_scatter_A',
            'l3_scatter_b',
            'l3_water_frac',
            'l3_BFi',
        )
        dtype = np.dtype([(k, 'f8') for k in fields])
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
            ydata = np.concatenate((self.rdCW[cw_slices].flat, self.ac[self.fd_slices].flat, self.phase[self.fd_slices].flat, self.g2[self.dcs_slices].flat))
            sigma = np.concatenate((self.rdCW[cw_slices].flat, self.ac[self.fd_slices].flatten() / 10, self.phase[self.fd_slices].flatten() / 10, self.g2[self.dcs_slices].flat))
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
            d_dcs_waves2 = np.cuda.to_device(dcs_waves * 1e-6)
            def residual(oparams):
                params = oparams.view(dtype)[0].copy()
                params['l1_water_frac'] /= 100
                params['l2_water_frac'] /= 100
                params['l3_water_frac'] /= 100
                params['l1_HbO_conc'] *= 1e-6
                params['l2_HbO_conc'] *= 1e-6
                params['l3_HbO_conc'] *= 1e-6
                params['l1_HbR_conc'] *= 1e-6
                params['l2_HbR_conc'] *= 1e-6
                params['l3_HbR_conc'] *= 1e-6
                params['l1_scatter_b'] /= 100
                params['l2_scatter_b'] /= 100
                params['l3_scatter_b'] /= 100
                cuda_calc_muN[ss_grid](params, d_cw_waves, d_cw_extc, d_cw_mua, d_cw_musp)
                cuda_calc_muN[fd_grid](params, d_fd_waves, d_fd_extc, d_fd_mua, d_fd_musp)
                cuda_calc_muN[dcs_grid](params, d_dcs_waves, d_dcs_extc, d_dcs_mua, d_dcs_musp)
                depths = np.array((params['l1'], params['l1'] + params['l2']))
                BFi = np.array((params['l1_BFi'], params['l2_BFi'], params['l3_BFi']))*1e-6
                cuda_model_nlayer_ss[ss_grid](d_cw_rhos, d_cw_mua, d_cw_musp, depths, n, n_ext, 20, 20, 1e-16, d_ss)
                cuda_model_nlayer_fd[fd_grid](d_fd_rhos, d_fd_mua, d_fd_musp, depths, freq, c, n, n_ext, 20, 20, 1e-16, d_fd)
                cuda_model_nlayer_g2[dcs_grid](d_dcs_rhos, d_tau, d_dcs_mua, d_dcs_musp, depths, BFi, d_dcs_waves2, n, n_ext, beta, tau_0, 20, 20, 1e-16, d_dcs)
                ss = d_ss.copy_to_host()
                fd = d_fd.copy_to_host()
                dcs = d_dcs.copy_to_host()
                fit = np.concatenate((ss, np.abs(fd) * params['ac_fit'], np.angle(fd), dcs))
                return (fit - ydata) / sigma
        else:
            pass
        return dtype, residual



"""
# det_slice = np.logical_or.reduce(np.equal.outer(rhos, (10,16,24,30)), axis=1)
det_slice = slice(20, None)
cw_slice = slice(None)
# fd_slice = slice(waves.size//7, None, waves.size//7)
fd_slice = np.logical_or.reduce(np.equal.outer(waves, (671, 692, 706, 727, 759, 784, 812, 830, 975)), axis=1)


n, n_ext, freq, c = 1.4, 1, 110e6, 2.998e11
beta = 0.5
first_tau_delay = 0

base_parameters = {
    'water_frac': {'value': 75, 'min': 50, 'max': 100},
    'HbO_conc': {'value': 60, 'min': 0, 'max': 100},
    'HbR_conc': {'value': 40, 'min': 0, 'max': 100},
    'scatter_A': {'value': 12, 'min': 0, 'max': 25},
    'scatter_b': {'value': 35, 'min': 10, 'max': 50},
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
    'l1_scatter_A': {'value': 8, 'min': 0, 'max': 25},
    'l2_scatter_A': {'value': 0.1, 'min': 0, 'max': 25},
    'l3_scatter_A': {'value': 12, 'min': 0, 'max': 25},
    'l1_scatter_b': {'value': 35, 'min': 10, 'max': 50},
    'l2_scatter_b': {'value': 35, 'min': 10, 'max': 50},
    'l3_scatter_b': {'value': 35, 'min': 10, 'max': 50},
    'l1_BFi': {'value': 1, 'min': 0, 'max': 20},
    'l2_BFi': {'value': 1e-3, 'min': 0, 'max': 20},
    'l3_BFi': {'value': 6, 'min': 0, 'max': 20},
    'l1': {'value': 3, 'min': 0, 'max': 6},
    'l2': {'value': 4, 'min': 0, 'max': 8},
}

fit_parameters = {
    'broadband': (
        'water_frac',
        'HbO_conc',
        'HbR_conc',
        'scatter_A',
        'scatter_b',
        'ac_fit'
    ),
    'broadband w/ dcs': (
        'water_frac',
        'HbO_conc',
        'HbR_conc',
        'scatter_A',
        'scatter_b',
        'ac_fit',
        'BFi'
    ),
    'time-domain': (
        'water_frac',
        'HbO_conc',
        'HbR_conc',
        'scatter_A',
        'scatter_b'
    ),
    'time-domain w/ dcs': (
        'water_frac',
        'HbO_conc',
        'HbR_conc',
        'scatter_A',
        'scatter_b',
        'BFi'
    ),
    '3L broadband': (
        'ac_fit',
        'l1',
        'l1_HbO_conc',
        'l1_HbR_conc',
        'l1_scatter_A',
        'l1_scatter_b',
        'l1_water_frac',
        'l2',
        'l2_HbO_conc',
        'l2_HbR_conc',
        'l2_scatter_A',
        'l2_scatter_b',
        'l2_water_frac',
        'l3_HbO_conc',
        'l3_HbR_conc',
        'l3_scatter_A',
        'l3_scatter_b',
        'l3_water_frac'
    ),
    '3L broadband w/ dcs': (
        'ac_fit',
        'l1',
        'l1_HbO_conc',
        'l1_HbR_conc',
        'l1_scatter_A',
        'l1_scatter_b',
        'l1_water_frac',
        'l1_BFi',
        'l2',
        'l2_HbO_conc',
        'l2_HbR_conc',
        'l2_scatter_A',
        'l2_scatter_b',
        'l2_water_frac',
        'l2_BFi',
        'l3_HbO_conc',
        'l3_HbR_conc',
        'l3_scatter_A',
        'l3_scatter_b',
        'l3_water_frac',
        'l3_BFi'
    ),
}


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


@jit
def create_mu_3LT(params, extc, waves):
    conc = np.array((
        (params['l1_water_frac'] / 100, params['l1_HbO_conc']*1e-6, params['l1_HbR_conc']*1e-6),
        (params['l2_water_frac'] / 100, params['l2_HbO_conc']*1e-6, params['l2_HbR_conc']*1e-6),
        (params['l3_water_frac'] / 100, params['l3_HbO_conc']*1e-6, params['l3_HbR_conc']*1e-6)
    ))
    mua = conc @ extc
    scA = np.array((params['l1_scatter_A'], params['l2_scatter_A'], params['l3_scatter_A']))
    scb = np.array((params['l1_scatter_b'], params['l2_scatter_b'], params['l3_scatter_b'])) / 100
    musp = scA.reshape((-1, 1)) * waves ** -scb.reshape((-1, 1))
    return mua, musp



def base_fit(name, compute, ydata, sigma):
    param_names = fit_parameters[name]
    params = [base_parameters[k] for k in param_names]
    init = np.array([p['value'] for p in params], dtype=np.float64)
    bounds = [p['min'] for p in params], [p['max'] for p in params]
    dtype = np.dtype([(f, 'float64') for f in parameters])
    def residual(params):
        fit = np.concatenate(compute(params.view(dtype)[0]))
        return (fit - ydata) / sigma
    #def chisq(params):
    #    return np.sum(residual(params) ** 2)
    opt = scipy.optimize.least_squares(fit_residual, init, bounds=bounds, verbose=0, max_nfev=None)
    return OrderedDict(zip(param_names, opt.x))


def broadband_fit(use_cuda, det_slice, cw_slice, fd_slice, rhos, waves, rdCW, ac, phase, n, n_ext, freq, c, **kwargs):
    cw_waves, cw_rhos = (v.flatten() for v in np.meshgrid(waves[cw_slice], rhos[det_slice], indexing='ij'))
    fd_waves, fd_rhos = (v.flatten() for v in np.meshgrid(waves[fd_slice], rhos[det_slice], indexing='ij'))
    cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR')
    fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR')
    ydata = np.concatenate((
        rdCW[cw_slice, det_slice].flat,
        ac[fd_slice, det_slice].flat,
        phase[fd_slice, det_slice].flat
    ))
    sigma = np.concatenate((
        rdCW[cw_slice, det_slice].flat,
        ac[fd_slice, det_slice].flatten() / 10,
        phase[fd_slice, det_slice].flatten() / 10
    ))
    if use_cuda:
        def compute(params):
            cw_mua, cw_musp = create_mu(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu(params, fd_extc, fd_waves)
            ss = np.zeros(cw_rhos.size, np.float64)
            fd = np.zeros(fd_rhos.size, np.complex128)
            cuda_model_ss[1 + ss.size // 64, 64](cw_rhos, cw_mua, cw_musp, n, n_ext, ss)
            cuda_model_fd[1 + fd.size // 64, 64](fd_rhos, fd_mua, fd_musp, n, n_ext, freq, c, fd)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd)
    else:
        @jit
        def compute(params):
            cw_mua, cw_musp = create_mu(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu(params, fd_extc, fd_waves)
            ss = model_ss(cw_rhos, cw_mua, cw_musp, n, n_ext)
            fd = model_fd(fd_rhos, fd_mua, fd_musp, n, n_ext, freq, c)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd)
    return base_fit('broadband', compute, ydata, sigma)


def broadband_dcs_fit(use_cuda, det_slice, cw_slice, fd_slice, rhos, waves, tau, rdCW, ac, phase, g2, n, n_ext, freq, c, beta, first_tau_delay, **kwargs):
    cw_waves, cw_rhos = (v.flatten() for v in np.meshgrid(waves[cw_slice], rhos[det_slice], indexing='ij'))
    fd_waves, fd_rhos = (v.flatten() for v in np.meshgrid(waves[fd_slice], rhos[det_slice], indexing='ij'))
    dcs_waves, dcs_rhos, dcs_tau = (v.flatten() for v in np.meshgrid(waves[fd_slice], rhos[det_slice], tau, indexing='ij'))
    cw_extc = get_extinction_coeffs(cw_waves, 'water', 'HbO', 'HbR')
    fd_extc = get_extinction_coeffs(fd_waves, 'water', 'HbO', 'HbR')
    dcs_extc = get_extinction_coeffs(dcs_waves, 'water', 'HbO', 'HbR')
    ydata = np.concatenate((
        rdCW[cw_slice, det_slice].flat,
        ac[fd_slice, det_slice].flat,
        phase[fd_slice, det_slice].flat,
        g2[fd_slice, det_slice].flat
    ))
    sigma = np.concatenate((
        rdCW[cw_slice, det_slice].flat,
        ac[fd_slice, det_slice].flatten() / 10,
        phase[fd_slice, det_slice].flatten() / 10,
        g2[fd_slice, det_slice].flat
    ))
    if use_cuda:
        def compute(params):
            cw_mua, cw_musp = create_mu(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu(params, fd_extc, fd_waves)
            dcs_mua, dcs_musp = create_mu(params, dcs_extc, dcs_waves)
            ss = np.zeros(cw_rhos.size, np.float64)
            fd = np.zeros(fd_rhos.size, np.complex128)
            dcs = np.zeros(dcs_rhos.size, np.float64)
            cuda_model_ss[1 + ss.size // 64, 64](cw_rhos, cw_mua, cw_musp, n, n_ext, ss)
            cuda_model_fd[1 + fd.size // 64, 64](fd_rhos, fd_mua, fd_musp, n, n_ext, freq, c, fd)
            cuda_model_g2[1 + dcs.size // 64, 64](tau, params['BFi']*1e-6, beta, dcs_mua, dcs_musp, dcs_waves*1e-6, dcs_rhos, first_tau_delay, n, dcs)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd), dcs
    else:
        @jit
        def compute(params):
            cw_mua, cw_musp = create_mu(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu(params, fd_extc, fd_waves)
            dcs_mua, dcs_musp = create_mu(params, dcs_extc, dcs_waves)
            ss = model_ss(cw_rhos, cw_mua, cw_musp, n, n_ext)
            fd = model_fd(fd_rhos, fd_mua, fd_musp, n, n_ext, freq, c)
            dcs = model_g2(tau, params['BFi']*1e-6, beta, dcs_mua, dcs_musp, dcs_waves*1e-6, dcs_rhos, first_tau_delay, n)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd), dcs
    return base_fit('broadband w/ dcs', compute, ydata, sigma)



def time_domain_fit(use_cuda, det_slice, time_slice, wave_slice, rhos, waves, tofs, rdTD, n, n_ext, c, **kwargs):
    td_waves, td_rhos, td_tof = (v.flatten() for v in np.meshgrid(waves[wave_slice], rhos[det_slice], tofs[time_slice], indexing='ij'))

    extc = get_extinction_coeffs(td_waves, 'water', 'HbO', 'HbR')
    ydata = rdTD[wave_slice][:, det_slice, time_slice].flatten() / dt
    sigma = ydata

    if use_cuda:
        def compute(params):
            mua, musp = create_mu(params, extc, waves)
            return model_td(time, rhos, mua, musp, n, n_ext, c)
    else:
        @jit
        def compute(params):
            mua, musp = create_mu(params, extc, waves)
            td = np.zeros(rhos.size, np.float64)
            cuda_model_td[1 + fit.size // 64, 64](time, rhos, mua, musp, n, n_ext, c, td)
            return td
    return base_fit('time-domain', compute, ydata, sigma)


def time_domain_dcs_fit(use_cuda, det_slice, time_slice, wave_slice, rhos, waves, tofs, rdTD, n, n_ext, c, **kwargs):
    td_waves, td_rhos, td_tof = (v.flatten() for v in np.meshgrid(waves[wave_slice], rhos[det_slice], tofs[time_slice], indexing='ij'))
    dcs_waves, dcs_rhos, dcs_tau = (v.flatten() for v in np.meshgrid(waves[wave_slice], rhos[det_slice], tau, indexing='ij'))

    td_extc = get_extinction_coeffs(td_waves, 'water', 'HbO', 'HbR')
    dcs_extc = get_extinction_coeffs(dcs_waves, 'water', 'HbO', 'HbR')
    
    ydata = np.concatenate((
        rdTD[wave_slice][:, det_slice, time_slice].flat,
        g2[wave_slice][:, det_slice].flat
    ))
    sigma = ydata

    if use_cuda:
        def compute(params):
            td_mua, td_musp = create_mu(params, td_extc, td_waves)
            dcs_mua, dcs_musp = create_mu(params, dcs_extc, dcs_waves)
            td = np.zeros(td_rhos.size, np.float64)
            dcs = np.zeros(dcs_rhos.size, np.float64)
            cuda_model_td[1 + td.size // 64, 64](tof, td_rhos, td_mua, td_musp, n, n_ext, c, td)
            cuda_model_g2[1 + dcs.size // 64, 64](tau, params['BFi']*1e-6, beta, dcs_mua, dcs_musp, dcs_waves*1e-6, dcs_rhos, first_tau_delay, n, dcs)
            return td, dcs
    else:
        @jit
        def compute(params):
            td_mua, td_musp = create_mu(params, td_extc, td_waves)
            dcs_mua, dcs_musp = create_mu(params, dcs_extc, dcs_waves)
            td = model_td(tof, td_rhos, td_mua, td_musp, n, n_ext, c)
            dcs = model_g2(tau, params['BFi']*1e-6, beta, dcs_mua, dcs_musp, dcs_waves*1e-6, dcs_rhos, first_tau_delay, n)
            return td, dcs
    return base_fit('time-domain w/ dcs', compute, ydata, sigma)


def broadband_3l_fit(use_cuda, **kwargs):

    if use_cuda:
        def compute(params):
            cw_mua, cw_musp = create_mu_3L(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu_3L(params, fd_extc, fd_waves)
            depths = np.array((params['l1'], params['l1'] + params['l2']))
            ss = np.zeros(cw_rhos.size, np.float64)
            fd = np.zeros(fd_rhos.size, np.complex128)    
            cuda_model_nlayer_ss[1 + ss.size // 64, 64](cw_rhos, cw_mua, cw_musp, depths, n, n_ext, 20, 20, 1e-16, ss)
            cuda_model_nlayer_fd[1 + fd.size // 64, 64](fd_rhos, fd_mua, fd_musp, depths, freq, c, n, n_ext, 20, 20, 1e-16, fd)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd)
    else:
        @jit
        def compute(params):
            cw_mua, cw_musp = create_mu_3LT(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu_3LT(params, fd_extc, fd_waves)
            depths = np.array((params['l1'], params['l1'] + params['l2']))
            ss = model_nlayer_ss(cw_rhos, cw_mua, cw_musp, depths, n, n_ext, 20, 20, 1e-16)
            fd = model_nlayer_fd(fd_rhos, fd_mua, fd_musp, depths, freq, c, n, n_ext, 20, 20, 1e-16)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd)
    return base_fit('3L broadband', compute, ydata, sigma)


def broadband_dcs_3l_fit(use_cuda, **kwargs):

    if use_cuda:
        def compute(params):
            cw_mua, cw_musp = create_mu_3L(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu_3L(params, fd_extc, fd_waves)
            dcs_mua, dcs_musp = create_mu_3L(params, dcs_extc, dcs_waves)
            depths = np.array((params['l1'], params['l1'] + params['l2']))
            BFi = np.array((params['l1_BFi'], params['l2_BFi'], params['l3_BFi'])) * 1e-6
            ss = np.zeros(cw_rhos.size, np.float64)
            fd = np.zeros(fd_rhos.size, np.complex128)
            dcs = np.zeros(dcs_rhos.size, np.float64)
            cuda_model_nlayer_ss[1 + ss.size // 64, 64](cw_rhos, cw_mua, cw_musp, depths, n, n_ext, 20, 20, 1e-16, ss)
            cuda_model_nlayer_fd[1 + fd.size // 64, 64](fd_rhos, fd_mua, fd_musp, depths, freq, c, n, n_ext, 20, 20, 1e-16, fd)
            cuda_model_nlayer_g2[1 + dcs.size // 64, 64](dcs_rhos, tau, dcs_mua, dcs_musp, depths, BFi, dcs_waves*1e-6, n, n_ext, beta, first_tau_delay, 20, 20, 1e-16, dcs)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd), dcs
    else:
        @jit
        def compute(params):
            cw_mua, cw_musp = create_mu_3LT(params, cw_extc, cw_waves)
            fd_mua, fd_musp = create_mu_3LT(params, fd_extc, fd_waves)
            dcs_mua, dcs_musp = create_mu_3LT(params, dcs_extc, dcs_waves)
            depths = np.array((params['l1'], params['l1'] + params['l2']))
            BFi = np.array((params['l1_BFi'], params['l2_BFi'], params['l3_BFi'])) * 1e-6
            ss = model_nlayer_ss(cw_rhos, cw_mua, cw_musp, depths, n, n_ext, 20, 20, 1e-16)
            fd = model_nlayer_fd(fd_rhos, fd_mua, fd_musp, depths, freq, c, n, n_ext, 20, 20, 1e-16)
            dcs = model_nlayer_g2(dcs_rhos, tau, dcs_mua, dcs_musp, depths, BFi, dcs_waves*1e-6, n, n_ext, beta, first_tau_delay, 20, 20, 1e-16)
            return ss, np.abs(fd) * params['ac_fit'], np.angle(fd), dcs
    return base_fit('3L broadband w/ dcs', compute, ydata, sigma)
"""