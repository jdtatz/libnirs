from collections.abc import Sequence
from dataclasses import asdict, dataclass

# FIXME: use actual `StrEnum` when min version is 3.11
from enum import Enum as StrEnum
from enum import IntFlag
from typing import Literal, NamedTuple, Optional, SupportsInt, TypedDict

import numpy as np
import numpy.typing as npt
from pmcx import run as _run  # , gpuinfo


class SaveFlags(IntFlag):
    DetectorId = 1
    NScatters = 2
    PartialPath = 4
    Momentum = 8
    ExitPosition = 16
    ExitDirection = 32
    InitialWeight = 64


_SAVE_FLAG_TO_CHAR_MAP = {
    SaveFlags.DetectorId: "D",
    SaveFlags.NScatters: "S",
    SaveFlags.PartialPath: "P",
    SaveFlags.Momentum: "M",
    SaveFlags.ExitPosition: "X",
    SaveFlags.ExitDirection: "V",
    SaveFlags.InitialWeight: "I",
}


class VolumetricOutputType(StrEnum):
    Flux = "flux"
    """Fluence rate [Power]/[Area]"""
    Fluence = "fluence"
    """Fluence [Energy]/[Area]"""
    Energy = "energy"
    """Voxel-wise energy deposit [Energy]"""
    Jacobian = "jacobian"
    """Jacobian for mua [Energy]/[Length]"""
    NScatter = "nscat"
    """Scattering count"""
    PartialMomentum = "wm"
    """Partial momentum transfer"""
    # TODO: `RF` & `DCS` are misparsed in pmcx


class DetectedPhotons(NamedTuple):
    detector_id: Optional[np.ndarray] = None
    nscatters: Optional[np.ndarray] = None
    partial_path: Optional[np.ndarray] = None
    momentum: Optional[np.ndarray] = None
    exit_position: Optional[np.ndarray] = None
    exit_direction: Optional[np.ndarray] = None
    initial_weight: Optional[np.ndarray] = None


class Medium(NamedTuple):
    mua: float
    mus: float
    g: float
    n: float


# Medium.dtype = np.dtype([(f, np.float32) for f in Medium._fields], align=True)


class uint3(NamedTuple):
    x: int = 0
    y: int = 0
    z: int = 0


# uint3.dtype = np.dtype([(f, np.uint32) for f in uint3._fields], align=True)


class float3(NamedTuple):
    x: float = 0
    y: float = 0
    z: float = 0


# float3.dtype = np.dtype([(f, np.float32) for f in float3._fields], align=True)


class float4(NamedTuple):
    x: float = 0
    y: float = 0
    z: float = 0
    w: float = 0


# float4.dtype = np.dtype([(f, np.float32) for f in float4._fields], align=True)


def _to_structured(saveflags, nmedia, detp):
    fields = {}
    if SaveFlags.DetectorId in saveflags:
        fields["detector_id"] = detp[0]
        detp = detp[1:]
    if SaveFlags.NScatters in saveflags:
        fields["nscatters"] = detp[:nmedia]
        detp = detp[nmedia:]
    if SaveFlags.PartialPath in saveflags:
        fields["partial_path"] = detp[:nmedia]
        detp = detp[nmedia:]
    if SaveFlags.Momentum in saveflags:
        fields["momentum"] = detp[:nmedia]
        detp = detp[nmedia:]
    if SaveFlags.ExitPosition in saveflags:
        fields["exit_position"] = detp[:3]
        detp = detp[3:]
    if SaveFlags.ExitDirection in saveflags:
        fields["exit_direction"] = detp[:3]
        detp = detp[3:]
    if SaveFlags.InitialWeight in saveflags:
        fields["initial_weight"] = detp[:3]
        detp = detp[3:]
    return DetectedPhotons(**fields)


class _MCXOutput(TypedDict):
    traj: npt.NDArray[np.float32]  # np.ndarray[Shape[Literal[6], Any], np.float32]
    seeds: npt.NDArray[np.uint8]  # np.ndarray[Shape[Any, Any], np.uint8]
    vol: npt.NDArray
    detp: npt.NDArray[np.float32]  # np.ndarray[Shape[Any, Any], np.float32]
    dref: npt.NDArray[np.float32]
    flux: npt.NDArray[np.float32]
    stat: dict
    prop: npt.NDArray[np.float32]  # np.ndarray[Shape[Literal[4], Any], np.float32]


class MCXValidationError(Exception):
    pass


@dataclass
class MCX:
    vol: npt.NDArray
    """Simulation volume of property indices"""
    prop: Sequence[Medium]
    """Medium properties {mua: [1/mm], mus: [1/mm], g: [unitless], n: [unitless]}"""
    tstart: float
    """Start time [seconds]"""
    tstep: float
    """Time step [seconds]"""
    tend: float
    """End time [seconds]"""
    issrcfrom0: Optional[bool] = True
    """Is the source position 0-indexed?"""
    nphoton: Optional[int] = None
    """Total simulated photon number"""
    # TODO: change to SupportsInt | npt.NDarray[np.uint8] when I add `SEED_FROM_FILE` support
    seed: Optional[SupportsInt] = None
    """Integer to seed MCX's PRNG"""
    nblocksize: Optional[int] = None
    nthread: Optional[int] = None
    maxdetphoton: Optional[int] = None
    """Max number of detected photons that are saved"""
    sradius: Optional[float] = None
    maxgate: Optional[int] = None  # FIXME: if `issave2pt` then this should default to `ceil((tend - tstart) / tstep)`
    respin: Optional[int] = None
    isreflect: Optional[bool] = None
    """Reflect at external boundaries?"""
    isref3: Optional[bool] = None
    isrefint: Optional[bool] = None
    """Reflect at internal boundaries?"""
    isnormalized: Optional[bool] = None
    autopilot: Optional[bool] = None
    """Optimally set `nblocksize` & `nthread`"""
    minenergy: Optional[float] = None
    unitinmm: Optional[float] = None
    """[grid unit] in [mm]"""
    printnum: Optional[int] = None
    voidtime: Optional[int] = None
    issaveref: Optional[bool] = None
    issaveexit: Optional[bool] = None
    ismomentum: Optional[bool] = None
    isspecular: Optional[bool] = None
    replaydet: Optional[int] = None
    faststep: Optional[bool] = None
    maxvoidstep: Optional[int] = None
    maxjumpdebug: Optional[int] = None
    gscatter: Optional[int] = None
    srcnum: Optional[int] = None
    omega: Optional[float] = None
    lambda_: Optional[float] = None
    srcpos: Optional[float3] = None  # Technically a float4, but mcx discards the fourth component
    """Source position vector [grid unit]"""
    srcdir: Optional[float4] = None
    """Source direction unit vector [grid unit]"""
    steps: Optional[float3] = None
    crop0: Optional[uint3] = None
    crop1: Optional[uint3] = None
    # TODO Totally change srctype, srcparam1, srcparam2, & srcpattern into diffrent
    # classes per enum to totally encapsulate the needed information
    srcparam1: Optional[float4] = None
    srcparam2: Optional[float4] = None
    srcpattern: Optional[npt.NDArray] = None
    srciquv: Optional[float] = None
    detpos: Optional[Sequence[float4]] = None
    """Detector vector positions and radii [grid unit]"""
    # polprop: np.ndarray[tuple[Any, 5], np.float32]
    srctype: Optional[
        Literal[
            "pencil",
            "isotropic",
            "cone",
            "gaussian",
            "planar",
            "pattern",
            "fourier",
            "arcsine",
            "disk",
            "fourierx",
            "fourierx2d",
            "zgaussian",
            "line",
            "slit",
            "pencilarray",
            "pattern3d",
            "hyperboloid",
            "diskarray",
        ]
    ] = None
    """Source type"""
    debuglevel: Optional[Literal["R", "M", "P"]] = None  # FIXME
    # session: str
    # invcdf: ???
    # shapes: ???
    # bc: ???
    # gpuid: int | Sequence[bool]
    # workload: ???
    # detphotons: Optional[npt.NDArray] = None  # only needed for replay

    def run(
        self,
        output_type: Optional[VolumetricOutputType] = None,
        save_det_flags: Optional[SaveFlags] = None,
    ):
        cfg = asdict(self)
        cfg = {k: v for k, v in cfg.items() if v is not None}

        lambda_ = cfg.pop("lambda_", None)
        if lambda_:
            cfg["lambda"] = lambda_

        cfg["prop"] = np.asarray(cfg["prop"], dtype=np.float32, order="F")
        if "detpos" in cfg:
            cfg["detpos"] = np.asarray(cfg["detpos"], dtype=np.float32, order="F")
        seed = cfg.pop("seed", None)
        if seed is not None and np.isscalar(seed):
            # TODO: need python 3.11 to prevent false negatives,
            #  assert isinstance(seed, typing.SupportsInt)
            cfg["seed"] = int(seed)  # type: ignore
        elif seed is not None:
            raise NotImplementedError("`SEED_FROM_FILE` support hasn't been added yet")

        if self.tend < self.tstart:
            raise MCXValidationError("Simulation time end must be >= to time start")
        if self.tstep <= 0:
            raise MCXValidationError("Simulation time step must be > 0")

        # if not self.issrcfrom0:
        #     cfg["srcpos"].x -= 1
        #     cfg["srcpos"].y -= 1
        #     cfg["srcpos"].z -= 1
        #     cfg["issrcfrom0"] = True
        #     if self.detpos is not None:
        #         for i in range(self.detpos.shape[0]):
        #             cfg["detpos"][i].x -= 1
        #             cfg["detpos"][i].y -= 1
        #             cfg["detpos"][i].z -= 1

        if output_type:
            cfg["outputtype"] = output_type.value
            cfg["maxgate"] = int(np.ceil((self.tend - self.tstart) / self.tstep))
            cfg["issave2pt"] = True
        else:
            cfg["issave2pt"] = False

        if save_det_flags:
            if self.detpos is None:
                raise MCXValidationError("Saving detectors is enabled, but detector positions is not initialized")
            cfg["savedetflag"] = ""
            for sflag, c in _SAVE_FLAG_TO_CHAR_MAP.items():
                if sflag in save_det_flags:
                    cfg["savedetflag"] += c
            cfg["issavedet"] = True
        else:
            cfg["issavedet"] = False

        output: _MCXOutput = _run(cfg)

        if save_det_flags:
            detp = _to_structured(save_det_flags, len(self.prop) - 1, output["detp"])
        else:
            detp = None

        if output_type:
            flux = output["flux"]
            stats = output["stat"]
        else:
            flux = None
            stats = None

        return detp, flux, stats, output
