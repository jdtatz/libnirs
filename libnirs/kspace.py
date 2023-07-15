""" kWave-Toolbox Wrapper http://www.k-wave.org """
from typing import NamedTuple, Union, Optional, Sequence, Tuple
from enum import IntEnum
from datetime import datetime
from subprocess import run, PIPE, STDOUT
from tempfile import TemporaryDirectory
from pathlib import Path

from h5py import File, Group
import numpy as np
from scipy import signal


def create_header(group: Group, name: str = "User", description: str = "short description"):
    group.attrs["created_by"] = np.string_(name)
    group.attrs["creation_date"] = np.string_(datetime.now().isoformat())
    group.attrs["file_description"] = np.string_(description)
    group.attrs["file_type"] = np.string_("input")
    group.attrs["major_version"] = np.string_("1")
    group.attrs["minor_version"] = np.string_("2")


def write_values(group: Group, values: dict):
    for k, v in values.items():
        if v is None:
            continue
        elif isinstance(v, bool):
            dst = group.create_dataset(name=k, shape=(1, 1, 1), dtype=np.uint64, data=(1 if v else 0))
            dst.attrs["data_type"] = np.string_("long")
        elif isinstance(v, int):
            dst = group.create_dataset(name=k, shape=(1, 1, 1), dtype=np.uint64, data=v)
            dst.attrs["data_type"] = np.string_("long")
        elif isinstance(v, float):
            dst = group.create_dataset(name=k, shape=(1, 1, 1), dtype=np.float32, data=v)
            dst.attrs["data_type"] = np.string_("float")
        elif isinstance(v, np.ndarray):
            if v.ndim == 1:
                v = v[:, np.newaxis, np.newaxis]
            elif v.ndim == 2:
                v = v[:, :, np.newaxis]
            v = v.T
            if issubclass(v.dtype.type, np.integer):
                dst = group.create_dataset(name=k, shape=v.shape, dtype=np.uint64, data=v)
                dst.attrs["data_type"] = np.string_("long")
            elif issubclass(v.dtype.type, np.floating):
                dst = group.create_dataset(name=k, shape=v.shape, dtype=np.float32, data=v)
                dst.attrs["data_type"] = np.string_("float")
            else:
                raise Exception(f"Unknown numpy val: {v} for key: {k}")
        else:
            raise Exception(f"Unknown val: {v} for key: {k}")
        dst.attrs["domain_type"] = np.string_("real")


def read_values(group: Group):
    export = {}
    for k, v in group.attrs.items():
        export[k] = v
    for k, v in group.items():
        v = v[()]
        if v.size == 1:
            v = v.item()
        else:
            v = np.transpose(v)
        export[k] = v
    return export


def matlab_compat_ravel_multi_index(*xyz, shape):
    return 1 + np.ravel_multi_index(xyz, shape, order='F')


class kGrid:
    def __init__(self, Nx, dx, Ny=None, dy=None, Nz=None, dz=None):
        self.Nx = Nx
        self.dx = dx
        if Ny is not None:
            self.Ny = Ny
            assert dy is not None
            self.dy = dy
            if Nz is not None:
                self.Nz = Nz
                assert dz is not None
                self.dz = dz
                self.ndim = 3
            else:
                assert dz is None
                self.ndim = 2
        else:
            assert dy is None
            assert Nz is None
            assert dz is None
            self.ndim = 1


def make_time(Nxyz: Sequence[int], dxyz: Sequence[float], sound_speed: float, cfl=0.3, t_end=None) -> Tuple[int, float]:
    # cfl = Courant–Friedrichs–Lewy condition
    if t_end is None:
        t_end = np.sqrt(sum((n * d)**2 for n, d in zip(Nxyz, dxyz))) / np.min(sound_speed)
    min_grid_dim = min(dxyz)
    dt = cfl * min_grid_dim / np.max(sound_speed)
    Nt = int(t_end // dt) + 1
    return Nt, dt


# Sensor mask type (linear or cuboid corners)
class SensorMaskType(IntEnum):
    # Linear sensor mask
    Index = 0
    # Cuboid corners sensor mask
    Corners = 1


# Source mode (Dirichlet, additive, additive-no-correction)
class SourceMode(IntEnum):
    # Dirichlet source condition
    Dirichlet = 0
    # Additive-no-correction source condition
    AdditiveNoCorrection = 1
    # Additive source condition
    Additive = 2


# Medium absorption type
class AbsorptionType(IntEnum):
    # No absorption
    Lossless = 0
    # Power law absorption
    PowerLaw = 1
    # Stokes absorption
    Stokes = 2


# Absorbing Medium Properties
class AbsorbingMediumProperties(NamedTuple):
    alpha_coef: Union[float, "NDArray[('Nx', 'Ny', 'Nz'), Float]"]
    alpha_power: float


# Pressure Source Properties
class PressureSourceProperties(NamedTuple):
    p_source_mode: SourceMode
    p_source_index: "NDArray['Nsrc', UInt]"
    p_source_input: Union["NDArray['Nt_src', Float]", "NDArray[('Nsrc', 'Nt_src'), Float]"]


# Velocity Source Properties
class VelocitySourceProperties(NamedTuple):
    u_source_mode: SourceMode
    u_source_index: "NDArray['Nsrc', UInt]"
    ux_source_input: Union["NDArray['Nt_src', Float]", "NDArray[('Nsrc', 'Nt_src'), Float]"]
    uy_source_input: Union["NDArray['Nt_src', Float]", "NDArray[('Nsrc', 'Nt_src'), Float]"]
    uz_source_input: Union["NDArray['Nt_src', Float]", "NDArray[('Nsrc', 'Nt_src'), Float]"]


class Parameters(NamedTuple):
    Nx: int
    Ny: int
    Nz: int
    Nt: int
    dx: float
    dy: float
    dz: Optional[float]
    dt: float
    # Sensor Variables
    sensor_mask_type: SensorMaskType
    sensor_mask: Union["NDArray['Nsensor', UInt]", "NDArray[('Ncubes', 6), UInt]"]

    # Medium Properties
    # use_staggered_grid: bool = False  # Not Yet Implmented
    density: Union[float, "NDArray[('Nx', 'Ny', 'Nz'), Float]"]
    c0: Union[float, "NDArray[('Nx', 'Ny', 'Nz'), Float]"]
    c_ref: Optional[float] = None
    # Nonlinear Medium Properties
    BonA: Optional[Union[float, "NDArray[('Nx', 'Ny', 'Nz'), Float]"]] = None
    # Absorbing Medium Properties
    absorption_type: AbsorptionType = AbsorptionType.Lossless
    absorbing_properties: Optional[AbsorbingMediumProperties] = None
    # Source Properties

    # (defined if (ux_source_flag == 1 || uy_source_flag == 1 || uz_source_flag == 1))
    u_source: Optional[VelocitySourceProperties] = None

    # (defined if (p_source_flag == 1))
    p_source: Optional[PressureSourceProperties] = None

    # (defined if (transducer_source_flag == 1))
    # u_source_index: Optional[Sequence[int]]
    # transducer_source_input: Optional[Sequence[float]]
    # delay_mask: Optional[Sequence[float]]

    # (defined if (p0_source_flag == 1))
    p0_source_input: Optional["NDArray[('Nx', 'Ny', 'Nz'), Float]"] = None

    # PML Variables
    pml_x_size: int = 10
    pml_y_size: int = 10
    pml_z_size: Optional[int] = 10
    pml_x_alpha: float = 2.0
    pml_y_alpha: float = 2.0
    pml_z_alpha: Optional[float] = 2.0

    def write(self, group: Group):
        values = {
            "ux_source_flag": 0 if self.u_source is None else self.u_source.ux_source_input.shape[-1],
            "uy_source_flag": 0 if self.u_source is None else self.u_source.uy_source_input.shape[-1],
            "uz_source_flag": 0 if self.u_source is None else self.u_source.uz_source_input.shape[-1],
            "p_source_flag": 0 if self.p_source is None else self.p_source.p_source_input.shape[-1],
            "p0_source_flag": self.p0_source_input is not None,
            "transducer_source_flag": False,
            "nonuniform_grid_flag": False,
            "nonlinear_flag": self.BonA is not None,
            "absorbing_flag": int(self.absorption_type),
            "axisymmetric_flag": False,
            "Nx": self.Nx,
            "Ny": self.Ny,
            "Nz": self.Nz,
            "Nt": self.Nt,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "dt": self.dt,
            "rho0": self.density,
            "rho0_sgx": self.density,
            "rho0_sgy": self.density,
            "rho0_sgz": self.density,
            "c0": self.c0,
            "c_ref": self.c_ref if self.c_ref is not None else np.max(self.c0),
            "BonA": self.BonA,
            "sensor_mask_type": int(self.sensor_mask_type),
            "p0_source_input": self.p0_source_input,
            "pml_x_size": self.pml_x_size,
            "pml_y_size": self.pml_y_size,
            "pml_z_size": self.pml_z_size,
            "pml_x_alpha": self.pml_x_alpha,
            "pml_y_alpha": self.pml_y_alpha,
            "pml_z_alpha": self.pml_z_alpha,
        }

        if self.sensor_mask_type == SensorMaskType.Index:
            values["sensor_mask_index"] = self.sensor_mask
        elif self.sensor_mask_type == SensorMaskType.Corners:
            values["sensor_mask_corners"] = self.sensor_mask
        else:
            raise Exception("Invalid SensorMaskType")

        if self.absorbing_properties is not None:
            assert self.absorption_type != AbsorptionType.Lossless
            values["alpha_coef"] = self.absorbing_properties.alpha_coef
            values["alpha_power"] = self.absorbing_properties.alpha_power

        if self.p_source is not None:
            source = self.p_source.p_source_input
            values["p_source_mode"] = int(self.p_source.p_source_mode)
            values["p_source_many"] = source.ndim == 2
            values["p_source_index"] = self.p_source.p_source_index
            if source.ndim == 1:
                source = source[np.newaxis, :]
            values["p_source_input"] = source

        if self.u_source is not None:
            source_ux = self.u_source.ux_source_input
            source_uy = self.u_source.uy_source_input
            source_uz = self.u_source.uz_source_input
            assert source_ux.shape == source_uy.shape and source_uy.shape == source_uz.shape
            values["u_source_mode"] = int(self.u_source.u_source_mode)
            values["u_source_many"] = source_ux.ndim == 2
            values["u_source_index"] = self.u_source.u_source_index
            if source_ux.ndim == 1:
                source_ux = source_ux[np.newaxis, :]
                source_uy = source_uy[np.newaxis, :]
                source_uz = source_uz[np.newaxis, :]
            values["ux_source_input"] = source_ux
            values["uy_source_input"] = source_uy
            values["uz_source_input"] = source_uz

        return write_values(group, values)


def run_k_space(
    params: Parameters,
    use_cpu: bool = False,
    name: str = "User",
    description: str = "short description",
    kspace_path = None,
):
    with TemporaryDirectory() as tdir:
        in_file = Path(tdir) / "input.hdf5"
        out_file = Path(tdir) / "output.hdf5"
        with File(in_file, "w") as f:
            create_header(f, name, description)
            params.write(f)
        if kspace_path is None:
            kspace_path = Path(__file__).parent / (
                "kspaceFirstOrder-OMP" if use_cpu else "kspaceFirstOrder-CUDA"
            )
        cmd = (
            kspace_path,
            "-i",
            in_file,
            "-o",
            out_file,
            "--copy_sensor_mask",
            "-p",
            "--p_final",
            "-u",
            "--u_final",
        )
        proc = run(cmd, stdout=PIPE, stderr=STDOUT)
        stdout = proc.stdout.decode("utf-8", "ignore")
        try:
            proc.check_returncode()
        except Exception as e:
            print(stdout)
            raise
        with File(out_file, "r") as f:
            return {**read_values(f), "stdout": stdout}
