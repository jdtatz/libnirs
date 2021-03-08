import numpy as np
from numba import vectorize
from .model import _model_ss, _model_ss_sig, _model_fd, _model_fd_sig, _model_td, _model_td_sig, _model_g1, _model_g1_sig, _model_g2, _model_g2_sig

cuda_model_ss = vectorize(_model_ss_sig, target="cuda")(_model_ss)
cuda_model_fd = vectorize(_model_fd_sig, target="cuda")(_model_fd)
# cuda_model_td = vectorize(_model_td_sig, target="cuda")(_model_td)
cuda_model_g1 = vectorize(_model_g1_sig, target="cuda")(_model_g1)
cuda_model_g2 = vectorize(_model_g2_sig, target="cuda")(_model_g2)
