import os
if 'CUDA_HOME' not in os.environ:
    if 'CUDA_PATH' not in os.environ:
        raise Exception('Either $CUDA_PATH or $CUDA_HOME must be defined')
    else:
        os.environ['CUDA_HOME'] = os.environ['CUDA_PATH']
import time
import math
import numpy as np
import numba as nb
import numba.cuda
import numba.cuda.random

import numba.extending
from llvmlite import ir
import operator


#  Constants and types for monte carlo simulation
it = np.int32
ft = np.float32
roulette_const = ft(10)
roulette_threshold = ft(1e-4)
spec_dtype = np.dtype([
    ('nphoton', it),
    ('srcpos', ft, 3),
    ('srcdir', ft, 3),
    ('tend', ft),
    ('tstep', ft),
    ('lightspeed', ft),
    ('isflu', 'b'),
    ('isdet', 'b')
])
state_dtype = np.dtype([
    ('mua', ft),
    ('mus', ft),
    ('g', ft),
    ('n', ft),
    ('BFi', ft),
])
uninit_boundry = it(-1)
x_boundry = it(0)
y_boundry = it(1)
z_boundry = it(2)


#  Numba-compatible Vector Type
class V3:
    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z
    
    def __add__(self, rhs):
        return V3(self.x + rhs, self.y + rhs, self.z + rhs)

    def __sub__(self, rhs):
        return V3(self.x - rhs, self.y - rhs, self.z - rhs)

    def __mul__(self, rhs):
        if isinstance(rhs, V3):
            return V3(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
        else:
            return V3(self.x * rhs, self.y * rhs, self.z * rhs)
    
    def __truediv__(self, rhs):
        return V3(self.x / rhs, self.y / rhs, self.z / rhs)

class V3Type(nb.types.Type):
    def __init__(self, scalar_type):
        super().__init__(name="V3({})".format(scalar_type))
        self.scalar_type = scalar_type
    
    def cast_python_value(self, value):
        return self.scalar_type(value)

@nb.extending.type_callable(V3)
def call_v3_typer(c):
    def v3_typer(x, y, z):
        assert x == y == z
        return V3Type(x)
    return v3_typer


@nb.extending.lower_builtin(V3, nb.types.Number, nb.types.Number, nb.types.Number)
def impl_v3(context, builder, sig, args):
    stype = args[0].type
    ltype = ir.VectorType(stype, 3)
    vec = ltype(ir.Undefined)
    for i, v in enumerate(args):
        i = ir.Constant(ir.IntType(32), i)
        vec = builder.insert_element(vec, v, i)
    return vec
    
@nb.extending.infer_getattr
class V3AttributeTemplate(nb.typing.templates.AttributeTemplate):
    key = V3Type

    def generic_resolve(self, typ, attr):
        if attr == 'x' or attr == 'y' or attr == 'z':
            return typ.scalar_type


@nb.extending.lower_getattr_generic(V3Type)
def get_attr_v3(context, builder, typ, val, attr):
    i = {'x': 0, 'y': 1, 'z': 2}.get(attr, None)
    if i is not None:
        i = ir.Constant(ir.IntType(32), i)
        return builder.extract_element(val, i)

'''
@nb.extending.lower_setattr_generic(V3Type)
def set_attr_v3(context, builder, sig, args, attr):
    i = {'x': 0, 'y': 1, 'z': 2}.get(attr, None)
    if i is not None:
        vec, val = args
        i = ir.Constant(ir.IntType(32), i)
        return builder.insert_element(vec, val, i)
'''

@nb.extending.register_model(V3Type)
class V3TypeModel(nb.extending.models.PrimitiveModel):
    def __init__(self, dmm, fe_type):
        stype = dmm.lookup(fe_type.scalar_type).get_value_type()
        ltype = ir.VectorType(stype, 3)
        super().__init__(dmm, fe_type, ltype)
ops = [
    (operator.add, operator.iadd, 'add', 'fadd'),
    (operator.sub, operator.isub, 'sub', 'fsub'),
    (operator.mul, operator.imul, 'mul', 'fmul'),
    (operator.truediv, operator.itruediv, 'div', 'fdiv'),
]

class V3OpTemplate(nb.typing.templates.FunctionTemplate):
    def apply(self, args, kwds):
        lhs, rhs = args
        if isinstance(lhs, V3Type):
            if isinstance(rhs, nb.types.Number) or isinstance(rhs, V3Type):
                return nb.typing.signature(lhs, lhs, rhs)

def create_op(op, iop, int_op, float_op):
    op_tmpl = type(f"{int_op}V3Template", (V3OpTemplate,), {'key': op})
    iop_tmpl = type(f"{int_op}InplaceV3Template", (V3OpTemplate,), {'key': iop})
    nb.extending.infer(op_tmpl)
    nb.extending.infer(iop_tmpl)
    nb.typing.templates.infer_global(op, nb.types.Function(op_tmpl))
    nb.typing.templates.infer_global(iop, nb.types.Function(iop_tmpl))
    
    @nb.extending.lower_builtin(op, V3Type, nb.types.Number)
    @nb.extending.lower_builtin(iop, V3Type, nb.types.Number)
    def lower_v3_scalar_op(context, builder, sig, args):
        ltype, rtype = sig.args
        lhs, rhs = args
        bop = getattr(builder, float_op if isinstance(ltype.scalar_type, numba.types.scalars.Float) else int_op)
        rhs = context.cast(builder, rhs, rtype, ltype.scalar_type)
        out = ir.VectorType(lhs.type.element, 3)(ir.Undefined)
        for i in range(3):
            i = ir.Constant(ir.IntType(32), i)
            v = builder.extract_element(lhs, i)
            v = bop(v, rhs)
            out = builder.insert_element(out, v, i)
        return out

    @nb.extending.lower_builtin(op, V3Type, V3Type)
    @nb.extending.lower_builtin(iop, V3Type, V3Type)
    def lower_v3_vec_op(context, builder, sig, args):
        ltype, rtype = sig.args
        lhs, rhs = args
        bop = getattr(builder, float_op if isinstance(ltype.scalar_type, numba.types.scalars.Float) else int_op)
        out = ir.VectorType(lhs.type.element, 3)(ir.Undefined)
        for i in range(3):
            i = ir.Constant(ir.IntType(32), i)
            lv = builder.extract_element(lhs, i)
            rv = builder.extract_element(rhs, i)
            rv = context.cast(builder, rv, rtype.scalar_type, ltype.scalar_type)
            v = bop(lv, rv)
            out = builder.insert_element(out, v, i)
        return out

for op, iop, int_op, float_op in ops:
    create_op(op, iop, int_op, float_op)



#  Monte Carlo Helper functions
@nb.cuda.jit(device=True)
def launch(spec, rng):
    """Launch a photon from the source"""
    x, y, z = spec.srcpos
    vx, vy, vz = spec.srcdir
    return V3(x, y, z), V3(vx, vy, vz)


@nb.cuda.jit(device=True)
def boundry_intersection(prev, vpos, vdir):
    """Find intersection of the traveling photon and the voxel its leaving.
    Then compute the distance from the photon's origin to the intersection.
    """
    def distance_to_boundry(which_boundry, voxel_pos, dir_cos):
        # Photon is not traveling along this axis
        if dir_cos == ft(0):
            return ft(0)
        # If the previous intersection was along this axis, then it must travel the entire voxel
        if prev == which_boundry:
            voxel_dist = ft(1)
        else:
            if dir_cos > ft(0):
                voxel_dist = ft(1) - voxel_pos
            else:
                voxel_dist = voxel_pos
        return ft(abs(voxel_dist / dir_cos))
    x_dist = distance_to_boundry(x_boundry, vpos.x, vdir.x)
    y_dist = distance_to_boundry(y_boundry, vpos.y, vdir.y)
    z_dist = distance_to_boundry(z_boundry, vpos.z, vdir.z)
    if x_dist != 0 and x_dist <= y_dist and x_dist <= z_dist:
        return x_dist, x_boundry
    elif y_dist != 0 and y_dist <= z_dist:
        return y_dist, y_boundry
    else:
        return z_dist, z_boundry


@nb.cuda.jit(nb.f4(nb.f4, nb.f4), device=True)
def henyey_greenstein_phase(g, rand):
    """The Henyey-Greenstein phase function, to compute a new pseudo random 
    direction of the photon after scattering
    """
    if g != 0:
        return (ft(1) / (ft(2) * g)) * (ft(1) + g ** 2 -((ft(1) - g ** 2) / (ft(1) - g + ft(2) * g * rand)) ** 2)
    else:
        return ft(1) - ft(2) * rand

@nb.cuda.jit(fastmath=False)
def monte_carlo(spec, states, media, rng, detpos, fluence, phi_td, phi_fd, freq, photon_counter, g1_top, tau, k0_sqr):
    buffer = nb.cuda.shared.array(0, nb.f4)
    nmedia = states.shape[0] - 1
    gid = nb.cuda.grid(1)
    tid = nb.cuda.threadIdx.x
    ppath_ind = tid*nmedia
    count = it(0)
    weight = ft(0)
    reset = True
    n_out = states[0].n
    while True:
        if reset:
            reset = False
            outofbounds = False
            count += it(1)
            if count >= spec.nphoton:
                return
            weight = ft(1)
            ppos, pdir = launch(spec, rng)
            mpos = V3(it(ppos.x), it(ppos.y), it(ppos.z))
            intersected_boundry = uninit_boundry
            nscat = it(0)
            t = ft(0)
            mid = media[mpos.x, mpos.y, mpos.z]
            state = states[mid]
            if n_out != state.n:
                weight *= ft(4)*state.n*n_out/(state.n+n_out)**2
            for j in range(nmedia):
                buffer[ppath_ind + j] = 0.0
        # move
        rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
        s = -math.log(rand) / (state.mua + state.mus)
        dist, intersected_boundry = boundry_intersection(intersected_boundry, ppos - mpos, pdir)
        while s > dist:
            ppos += pdir * dist
            t += dist * state.n / spec.lightspeed
            s -= dist
            if mid > 0:
                buffer[ppath_ind + mid - 1] += dist
            if intersected_boundry == x_boundry:
                if pdir.x >= 0:
                    mpos = V3(it(mpos.x + it(1)), mpos.y, mpos.z)
                else:
                    mpos = V3(it(mpos.x - it(1)), mpos.y, mpos.z)
                if not (it(0) <= mpos.x < it(media.shape[0])):
                    outofbounds = True
                    s = ft(0)
                    break
            elif intersected_boundry == y_boundry:
                if pdir.y >= 0:
                    mpos = V3(mpos.x, it(mpos.y + it(1)), mpos.z)
                else:
                    mpos = V3(mpos.x, it(mpos.y - it(1)), mpos.z)
                if not (it(0) <= mpos.y < it(media.shape[1])):
                    outofbounds = True
                    s = ft(0)
                    break
            else:
                if pdir.z >= 0:
                    mpos = V3(mpos.x, mpos.y, it(mpos.z + it(1)))
                else:
                    mpos = V3(mpos.x, mpos.y, it(mpos.z - it(1)))
                if not (it(0) <= mpos.z < it(media.shape[2])):
                    outofbounds = True
                    s = ft(0)
                    break
            mid, old_mid = media[mpos.x, mpos.y, mpos.z], mid
            if mid == 0:
                break
            if mid != old_mid:
                old_mut = (state.mua + state.mus)
                state = states[mid]
                s *= old_mut / (state.mua + state.mus)
            dist, intersected_boundry = boundry_intersection(intersected_boundry, ppos - mpos, pdir)
        ppos += pdir * s
        t += s * state.n / spec.lightspeed
        if mid > 0:
            buffer[ppath_ind + mid - 1] += s
        if outofbounds or mid == 0 or t > spec.tend:
            if spec.isdet:
                for i in range(detpos.shape[0]):
                    dist = (detpos[i, 0] - ppos.x)**2 + (detpos[i, 1] - ppos.y)**2 + (detpos[i, 2] - ppos.z)**2
                    if detpos[i, 3]**2 < dist and dist < detpos[i, 4]**2:
                        opl = np.float32(0)  # optical path length
                        clog_fd = np.complex64(0)
                        g1_prep = np.float32(0)
                        for j in range(nmedia):
                            opl += (buffer[ppath_ind+j]) * (-states[1 + j].mua)
                            clog_fd += buffer[ppath_ind+j] * (-states[1 + j].mua - 2j * np.float32(math.pi) * freq * states[1 + j].n / spec.lightspeed)
                            # k = n k0 = 2 𝜋 n / λ  ; wavenumber of light in the media
                            # BFi = blood flow index = α D_b
                            # ⟨Δr^2(τ)⟩ = 6 D_b τ  => ⟨Δr^2(τ)⟩ = 6 α D_b τ = 6 BFi τ
                            # g1(τ) = 1/N Sum_i^N { exp[ Sum_j^nmedia { -(1/3) Y k^2 ⟨Δr^2(τ)⟩ } + opl ] }
                            g1_prep += buffer[ppath_ind+j] * (-np.float32(2) * k0_sqr * states[1 + j].n**2 * states[1 + j].BFi)
                        time_id = min(it(t // spec.tstep), phi_td.shape[2] - 1)
                        phi_td[gid, i, time_id] += np.exp(opl)
                        phi_fd[gid, i] += np.exp(clog_fd)
                        for j in range(len(tau)):
                            g1_top[gid, i, j] += np.exp(g1_prep * tau[j] + opl)
                        photon_counter[gid, i, time_id] += 1
                        break
            reset = True
            continue
        # absorb
        delta_weight = weight * state.mua / (state.mua + state.mus)
        if spec.isflu:
            # nb.cuda.atomic.add(fluence, (ix, iy, iz), delta_weight)
            nb.cuda.atomic.add(fluence, (mpos.x, mpos.y, mpos.z, it(t//spec.tstep)), delta_weight)
        weight -= delta_weight
        # scatter
        rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
        ct = henyey_greenstein_phase(state.g, rand)
        st = math.sqrt(ft(1) - ct**2)
        rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
        phi = ft(2 * math.pi) * rand
        sp = math.sin(phi)
        cp = math.cos(phi)
        if abs(pdir.z) < ft(1 - 1e-6):
            vx, vy, vz = pdir.x, pdir.y, pdir.z
            denom = math.sqrt(ft(1) - vz**2)
            pdir = V3(st*(vx*vz*cp-vy*sp) / denom + vx*ct, st*(vy*vz*cp+vx*sp) / denom + vy*ct, -denom*st*cp + vz*ct)
        else:
            vx, vy, vz = pdir.x, pdir.y, pdir.z
            pdir = V3(st * cp, st * sp, ct * math.copysign(ft(1), vz))
        nscat += it(1)
        # roulette
        if weight < roulette_threshold:
            rand = nb.cuda.random.xoroshiro128p_uniform_float32(rng, gid)
            reset = rand > ft(1 / roulette_const)
            weight *= roulette_const