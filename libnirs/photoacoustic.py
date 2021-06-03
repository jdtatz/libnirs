import numpy as np
from typing import Tuple, Union
from .kspace import make_time, Parameters, AbsorptionType, SensorMaskType, run_k_space, PressureSourceProperties, VelocitySourceProperties, SourceMode

def _is_scalar_or_shape(value, shape: Tuple[int, ...]) -> bool:
    return np.shape(value) == () or np.shape(value) == shape


def fluence_2_pressure(cw_fluence: np.ndarray, mua_map: Union[float, np.ndarray], gruneisen: Union[float, np.ndarray] = 0.12) -> np.ndarray:
    """ Convert CW-Absorption to Pressure
    Parameters:
        cw_fluence: [ J / mm^2 ]
        mua_map: [ 1 / mm ]
        gruneisen: [], Grüneisen parameter Γ representing the thermoelastic efficiency of the medium
    Returns:
        initial_pressure_image: [ Pa ]

    Sources:
        Photoacoustic measurement of the Grüneisen parameter of tissue
            doi: 10.1117/1.JBO.19.1.017007
        Temperature-dependent optoacoustic response and transient through zero Grüneisen parameter in optically contrasted media
            doi: 10.1016/j.pacs.2017.06.002
    """
    media_shape = cw_fluence.shape
    assert _is_scalar_or_shape(mua_map, media_shape)
    assert _is_scalar_or_shape(gruneisen, media_shape)
    # 1 J / mm^3 == 1e9 Pa
    initial_pressure = mua_map * gruneisen * 1e9 * cw_fluence # Pa
    return initial_pressure


def forward_simulation(pressure_image: np.ndarray, sound_speed: Union[float, np.ndarray], density: Union[float, np.ndarray], detpos: np.ndarray, voxel_size: Tuple[float, float, float] = (1, 1, 1)) -> Parameters:
    Nx, Ny, Nz = pressure_image.shape
    dx, dy, dz = voxel_size  # [mm]
    dx *= 1e-3  # [m]
    dy *= 1e-3  # [m]
    dz *= 1e-3  # [m]
    c_ref = np.max(sound_speed)
    Nt, dt = make_time((Nx, Ny, Nz), (dx, dy, dz), c_ref)

    # detpos2 = np.stack([r + (x, y, z) for x in [-1, 0, 1] for y in [-1, 0, 1] for z in [-1, 0, 1] for r in detpos.astype(np.int32)])
    # det_mask = 1 + np.ravel_multi_index((detpos2.T).astype(np.int32), pressure_image.shape, order='F')
    det_mask = 1 + np.ravel_multi_index((detpos.T).astype(np.int32), pressure_image.shape, order='F')

    return Parameters(
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        Nt=Nt,
        dx=dx,
        dy=dy,
        dz=dz,
        dt=dt,
        density=density,
        c0=sound_speed,
        c_ref=c_ref,
        BonA=None,
        absorption_type=AbsorptionType.Lossless,
        absorbing_properties=None,
        sensor_mask_type=SensorMaskType.Index,
        sensor_mask=det_mask,
        p_source=None,
        u_source=None,
        p0_source_input=pressure_image,
        pml_x_size=10,
        pml_y_size=10,
        pml_z_size=10,
        pml_z_alpha=2.0,
    )


def time_reversed_simulation(forward_sim_parameters: Parameters, forward_sim_result: dict) -> Parameters:
    time_rev_p_source = PressureSourceProperties(
        p_source_mode=SourceMode.Dirichlet,
        p_source_index=forward_sim_parameters.sensor_mask,
        p_source_input=forward_sim_result["p"][:, ::-1, 0]
    )
    time_rev_u_source = VelocitySourceProperties(
        u_source_mode=SourceMode.Dirichlet,
        u_source_index=forward_sim_parameters.sensor_mask,
        ux_source_input=forward_sim_result["ux"][:, ::-1, 0],
        uy_source_input=forward_sim_result["uy"][:, ::-1, 0],
        uz_source_input=forward_sim_result["uz"][:, ::-1, 0],
    )
    return forward_sim_parameters._replace(
        p0_source_input=None,
        p_source=time_rev_p_source,
        u_source=time_rev_u_source
    )
