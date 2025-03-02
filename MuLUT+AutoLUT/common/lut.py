import torch
from torch import nn, Tensor
from torch.nn import functional as F

from jaxtyping import jaxtyped, Float, Int64 as Long, Int8, Int64
from typeguard import typechecked as typechecker
from typing import Any, Self, Tuple, List, Dict
import types

class LUT_unused(nn.Module):
    def __init__(self, interval, upscale):
        super().__init__()
        self.interval=interval
        self.upscale=upscale
        self.q=2**interval
        self.L=2**(8-interval)+1
        self.weight=nn.Parameter(torch.ones(self.L, self.L, self.L, self.L, upscale, upscale))

    p_val_type=Int8[Tensor, "batch channel ch cw"]
    def __interpolate_colors(
        self: Self,
        p: types.GenericAlias(tuple, ()),
        f
    ):
        """Interpolate colors

        Args:
            p (Tuple): p0000, p0001, ..., p1111     shape=(img_a1.shape[0], img_a1.shape[1], img_a1.shape[2], img_a1.shape[3], upscale, upscale)
            f (Tuple): fa, fb, fc, fd
        """
        p0000, p0001, p0010, p0011, p0100, p0101, p0110, p0111, p1000, p1001, p1010, p1011, p1100, p1101, p1110, p1111 = p
        fa, fb, fc, fd = f
        q=self.q

        pshape=p0000.shape
        target_shape=(*p0000.shape, self.upscale, self.upscale)
        out=torch.zeros(target_shape, dtype=self.weight.dtype).to(self.weight.device)
        sz = pshape[0] * pshape[1] * pshape[2] * pshape[3]
        out = out.reshape(sz, -1)

        p0000 = p0000.reshape(sz, -1)
        p0100 = p0100.reshape(sz, -1)
        p1000 = p1000.reshape(sz, -1)
        p1100 = p1100.reshape(sz, -1)
        fa = fa.reshape(-1, 1)

        p0001 = p0001.reshape(sz, -1)
        p0101 = p0101.reshape(sz, -1)
        p1001 = p1001.reshape(sz, -1)
        p1101 = p1101.reshape(sz, -1)
        fb = fb.reshape(-1, 1)
        fc = fc.reshape(-1, 1)

        p0010 = p0010.reshape(sz, -1)
        p0110 = p0110.reshape(sz, -1)
        p1010 = p1010.reshape(sz, -1)
        p1110 = p1110.reshape(sz, -1)
        fd = fd.reshape(-1, 1)

        p0011 = p0011.reshape(sz, -1)
        p0111 = p0111.reshape(sz, -1)
        p1011 = p1011.reshape(sz, -1)
        p1111 = p1111.reshape(sz, -1)

        fab = fa > fb;
        fac = fa > fc;
        fad = fa > fd

        fbc = fb > fc;
        fbd = fb > fd;
        fcd = fc > fd

        i1 = i = torch.all(torch.cat([fab, fbc, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i2 = i = torch.all(torch.cat([~i1[:, None], fab, fbc, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fb[i]) * p1000[i] + (fb[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i3 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], fab, fbc, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i4 = i = torch.all(torch.cat([~i1[:, None], ~i2[:, None], ~i3[:, None], fab, fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fb[i]) * p1001[i] + (fb[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i5 = i = torch.all(torch.cat([~(fbc), fab, fac, fbd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i6 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], fab, fac, fcd], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fc[i]) * p1000[i] + (fc[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i7 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], fab, fac, fad], dim=1), dim=1)
        out[i] = (q - fa[i]) * p0000[i] + (fa[i] - fd[i]) * p1000[i] + (fd[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i8 = i = torch.all(torch.cat([~(fbc), ~i5[:, None], ~i6[:, None], ~i7[:, None], fab, fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fa[i]) * p0001[i] + (fa[i] - fc[i]) * p1001[i] + (fc[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i9 = i = torch.all(torch.cat([~(fbc), ~(fac), fab, fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fb[i]) * p1010[i] + (fb[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i10 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], fab, fad], dim=1), dim=1)  # c > a > d > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fa[i]) * p0010[i] + (fa[i] - fd[i]) * p1010[i] + (fd[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i11 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], fab, fcd], dim=1),
                            dim=1)  # c > d > a > b
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]
        i12 = i = torch.all(torch.cat([~(fbc), ~(fac), ~i9[:, None], ~i10[:, None], ~i11[:, None], fab], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fa[i]) * p0011[i] + (fa[i] - fb[i]) * \
                 p1011[i] + (fb[i]) * p1111[i]

        i13 = i = torch.all(torch.cat([~(fab), fac, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fc[i]) * p1100[i] + (fc[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i14 = i = torch.all(torch.cat([~(fab), ~i13[:, None], fac, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fa[i]) * p0100[i] + (fa[i] - fd[i]) * p1100[i] + (fd[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i15 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], fac, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]
        i16 = i = torch.all(torch.cat([~(fab), ~i13[:, None], ~i14[:, None], ~i15[:, None], fac], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fa[i]) * p0101[i] + (fa[i] - fc[i]) * \
                 p1101[i] + (fc[i]) * p1111[i]

        i17 = i = torch.all(torch.cat([~(fab), ~(fac), fbc, fad], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i18 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], fbc, fcd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fc[i]) * p0100[i] + (fc[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i19 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], fbc, fbd], dim=1), dim=1)
        out[i] = (q - fb[i]) * p0000[i] + (fb[i] - fd[i]) * p0100[i] + (fd[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i20 = i = torch.all(torch.cat([~(fab), ~(fac), ~i17[:, None], ~i18[:, None], ~i19[:, None], fbc], dim=1), dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fb[i]) * p0001[i] + (fb[i] - fc[i]) * p0101[i] + (fc[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        i21 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), fad], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fa[i]) * p0110[i] + (fa[i] - fd[i]) * \
                 p1110[i] + (fd[i]) * p1111[i]
        i22 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], fbd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fb[i]) * p0010[i] + (fb[i] - fd[i]) * p0110[i] + (fd[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i23 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], fcd], dim=1), dim=1)
        out[i] = (q - fc[i]) * p0000[i] + (fc[i] - fd[i]) * p0010[i] + (fd[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]
        i24 = i = torch.all(torch.cat([~(fab), ~(fac), ~(fbc), ~i21[:, None], ~i22[:, None], ~i23[:, None]], dim=1),
                            dim=1)
        out[i] = (q - fd[i]) * p0000[i] + (fd[i] - fc[i]) * p0001[i] + (fc[i] - fb[i]) * p0011[i] + (fb[i] - fa[i]) * \
                 p0111[i] + (fa[i]) * p1111[i]

        out = out.reshape(target_shape)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(target_shape)
        out = out / q
        return out

class LUTWeight(nn.Module):
    def __init__(self, interval, upscale):
        super().__init__()
        self.interval=interval
        self.upscale=upscale
        self.q=2**interval
        self.L=2**(8-interval)+1
        self.init_weights()

    def init_weights(self):
        self.weight=nn.Parameter(torch.ones(self.L, self.L, self.L, self.L, self.upscale, self.upscale))

    def __getitem__(self, index):
        return self.weight[index]

class DFC_LUT(LUTWeight):
    def init_weights(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        raise NotImplementedError()
        L=self.L
        d=index%L
        index=index//L
        c=index%L
        index=index//L
        b=index%L
        index=index//L
        a=index

        x,y,z,t = a,b,c,d
        index_flag_xy = (torch.abs(x - y) <= self.d)
        index_flag_xz = (torch.abs(x - z) <= self.d)
        index_flag_xt = (torch.abs(x - t) <= self.d)
        index_flag = (index_flag_xy & index_flag_xz) & index_flag_xt

class LUT_Symmetric(LUTWeight):
    def init_weights(self):
        # type_map: Map from pattern (rank of four pixels) to type
        type_map = torch.zeros((4,4,4,4), dtype=torch.long)+10
        rot_map=torch.zeros_like(type_map)+10   # How to rotate
        flip_map=torch.zeros_like(type_map)+10  # and flip to become the requested pattern
        self.register_buffer("type_map", type_map)
        self.register_buffer("rot_map", rot_map)
        self.register_buffer("flip_map", flip_map)
        self.gen_type_mapping(torch.Tensor([0,1,2,3]).reshape(2,2).long(), 0)
        self.gen_type_mapping(torch.Tensor([0,1,3,2]).reshape(2,2).long(), 1)
        self.gen_type_mapping(torch.Tensor([0,2,3,1]).reshape(2,2).long(), 2)

        q=self.q=2**self.interval
        L=self.L=2**(8-self.interval)+1

        # idx_map: Map from normal LUT index to symmetric LUT index
        idx_map=torch.zeros((L,L,L,L), dtype=torch.int32)-1
        idx=0
        for a in range(L):
            for b in range(L):
                for c in range(L):
                    for d in range(L):
                        if a<=b<=c<=d:
                            assert idx_map[a,b,c,d]==-1
                            idx_map[a,b,c,d]=idx
                            idx+=1
        self.register_buffer("idx_map", idx_map)

        self.w0=nn.Parameter(torch.zeros(idx, 1, self.upscale, self.upscale))
        self.w1=nn.Parameter(torch.zeros(idx, 1, self.upscale, self.upscale))
        self.w2=nn.Parameter(torch.zeros(idx, 1, self.upscale, self.upscale))

    @property
    def dtype(self): return self.w0.data.dtype

    @property
    def device(self): return self.w0.data.device

    @jaxtyped(typechecker=typechecker)
    def gen_type_mapping(
        self: Self,
        pattern: Long[Tensor, "2 2"],
        ptype: int
    ):
        for flip_idx in range(2):
            for rot_idx in range(4):
                rot=pattern.rot90(rot_idx).reshape(-1)
                assert self.type_map[*rot]==self.rot_map[*rot]==self.flip_map[*rot]==10
                self.type_map[*rot]=ptype
                self.rot_map[*rot]=rot_idx
                self.flip_map[*rot]=flip_idx
            pattern=pattern.flip([1])

    @jaxtyped(typechecker=typechecker)
    def sort_with_rank(self: Self, sampled: Int8[Tensor, "bcl 1 2 2"]) -> Tuple[
        Int8[Tensor, "bcl 1 2 2"],
        Long[Tensor, "bcl 1 2 2"],
    ]:
        windows = sampled.reshape(-1, 4)
        n=windows.shape[0]
        wsorted, indexes = torch.sort(windows, dim=-1)
        ranks = torch.empty_like(indexes, dtype=torch.long).to(sampled.device).scatter_(-1, indexes, torch.arange(4).repeat(n, 1).to(sampled.device))
        return wsorted.view(sampled.shape), ranks.view(-1, 1, 2, 2)

    @jaxtyped(typechecker=typechecker)
    def query_pattern(self: Self, x: Int8[Tensor, "bcl 1 2 2"]) -> Tuple[
        Int8[Tensor, "bcl 1 2 2"],
        Long[Tensor, "bcl 1 1 1"],
        Long[Tensor, "bcl 1 1 1"],
        Long[Tensor, "bcl 1 1 1"],
    ]:
        xsorted, ranks=self.sort_with_rank(x)
        ranks=ranks.view(-1,4)
        a,b,c,d = ranks[:,0], ranks[:,1], ranks[:,2], ranks[:,3]

        types=self.type_map[a,b,c,d].view(-1, 1, 1, 1)
        assert (types!=10).all()

        rot_idxs=self.rot_map[a,b,c,d].view(-1, 1, 1, 1)
        assert (rot_idxs!=10).all()

        flip_idxs=self.flip_map[a,b,c,d].view(-1, 1, 1, 1)
        assert (flip_idxs!=10).all()

        return xsorted, types, rot_idxs, flip_idxs

    @staticmethod
    def round_func(input):
        # Backward Pass Differentiable Approximation (BPDA)
        # This is equivalent to replacing round function (non-differentiable)
        # with an identity function (differentiable) only when backward
        forward_value = torch.round(input)
        out = input.clone()
        out.data = forward_value.data
        return out

    @jaxtyped(typechecker=typechecker)
    def __getitem__(self, index: Int64[Tensor, "num_of_pixel"]) -> Float[Tensor, "num_of_pixel 1 upscale upscale"]:
        # Extract four pixels
        L=self.L
        d=(index%L).view(-1,1,1,1);  index=index//L
        c=(index%L).view(-1,1,1,1);  index=index//L
        b=(index%L).view(-1,1,1,1);  index=index//L
        a=(index).view(-1,1,1,1)
        n=a.shape[0]

        # Get pattern type
        upper=torch.cat([a,b],dim=-1)
        lower=torch.cat([c,d], dim=-1)
        abcd=torch.cat([upper, lower], dim=-2).to(torch.int8)
        sabcd, ptype, rot_idxs, flip_idxs=self.query_pattern(abcd)
        sabcd=sabcd.long()
        prtype=ptype.view(-1,1,1,1).repeat(1, 1, self.upscale, self.upscale)

        # Get index from pixel value
        # sorted a,b,c,d
        sa, sb, sc, sd = sabcd[:,:,0,0].view(n,1,1,1), sabcd[:,:,0,1].view(n,1,1,1), sabcd[:,:,1,0].view(n,1,1,1), sabcd[:,:,1,1].view(n,1,1,1)
        idx = self.idx_map[sa, sb, sc, sd]  # shape: n,1,1,1

        # Get LUT value according to type
        out=torch.zeros((n, 1, self.upscale, self.upscale), dtype=self.w0.dtype).to(self.w0.device)

        c0=(ptype==0)
        c1=(ptype==1)
        c2=(ptype==2)

        # We define that w0[idx] is the network output for initial pattern fed to self.gen_type_mapping e.g. [0,2,3,1]
        # To get the network output for the requested pixels, we need to do the same transformations (rotating and flipping)
        # when we first got the pattern in self.type_map

        if c0.any(): out += c0 * self.w0[idx].view(n,1,self.upscale,self.upscale)
        if c1.any(): out += c1 * self.w1[idx].view(n,1,self.upscale,self.upscale)
        if c2.any(): out += c2 * self.w2[idx].view(n,1,self.upscale,self.upscale)

        # Put them back to right positions
        if self.upscale>1:
            assert (0<=flip_idxs).all() and (flip_idxs<=1).all()
            assert (0<=rot_idxs).all() and (rot_idxs<=3).all()
            flip_idxs=flip_idxs.repeat(1,1,self.upscale,self.upscale)
            rot_idxs=rot_idxs.repeat(1,1,self.upscale,self.upscale)

            cf=flip_idxs==1
            r1=rot_idxs==1
            r2=rot_idxs==2
            r3=rot_idxs==3

            if cf.any(): out = torch.flip(out, [3]) * cf + out * cf.logical_not()

            # k=0, do nothing
            if r1.any(): out = torch.rot90(out, k=1, dims=[2,3]) * r1 + out * r1.logical_not()
            if r2.any(): out = torch.rot90(out, k=2, dims=[2,3]) * r2 + out * r2.logical_not()
            if r3.any(): out = torch.rot90(out, k=3, dims=[2,3]) * r3 + out * r3.logical_not()

        return self.round_func(out).clamp(-128,127)

if __name__=='__main__':
    torch.manual_seed(6546.2)
    interval=4
    L = 2 ** (8 - interval) + 1

    x=LUT_Symmetric(interval,4).cuda()
    opt=torch.optim.Adam(x.parameters())

    def lookup(sampled):
        sampled=sampled.to(torch.int64)
        n, c, _h, _w = sampled.shape
        a,b,c,d=sampled[:,:,0,0].view(n,c,1,1),sampled[:,:,0,1].view(n,c,1,1),sampled[:,:,1,0].view(n,c,1,1),sampled[:,:,1,1].view(n,c,1,1)
        idx=a*L**3+b*L**2+c*L+d
        return x[idx.flatten()]

    sampled1=torch.Tensor([14,2,4,7]).view(1,1,2,2).repeat(2,1,1,1).to(torch.int8).cuda()
    y1=lookup(sampled1)

    sampled2=sampled1.rot90(k=1, dims=[2,3])
    y2=lookup(sampled2)
    assert (y2==y1.rot90(k=1, dims=[2,3])).all()

    sampled3=sampled1.flip([3])
    y3=lookup(sampled3)
    assert (y3==y1.flip([3])).all()

    sampled4=sampled1.flip([2])
    y4=lookup(sampled4)
    assert (y4==y1.flip([2])).all()

    sampled5=sampled3.flip([2])
    y5=lookup(sampled5)
    assert (y5==y3.flip([2])).all()

    sampled6=sampled5.rot90(k=1, dims=[2,3])
    y6=lookup(sampled6)
    assert (y6==y5.rot90(k=1, dims=[2,3])).all()

