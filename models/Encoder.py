
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import math
from argparse import Namespace
from typing import Optional, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from models import register
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Sobelx(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(Sobelx, self).__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        self.sobelx = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobelx.weight.data[0, 0, 0, 0] = 1.0
        self.sobelx.weight.data[0, 0, 1, 0] = 2.0
        self.sobelx.weight.data[0, 0, 2, 0] = 1.0

        self.sobelx.weight.data[0, 0, 0, 1] = 0.0
        self.sobelx.weight.data[0, 0, 1, 1] = 0.0
        self.sobelx.weight.data[0, 0, 2, 1] = 0.0

        self.sobelx.weight.data[0, 0, 0, 2] = -1.0
        self.sobelx.weight.data[0, 0, 1, 2] = -2.0
        self.sobelx.weight.data[0, 0, 2, 2] = -1.0
        self.sobelx.weight.requires_grad = False

        self.conv_after_sobelx = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
    def forward(self, x):
        y = self.sobelx(x)
        y = self.conv_after_sobelx(y)
        return y

class Sobely(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(Sobely, self).__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
       
        self.sobely = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobely.weight.data[0, 0, 0, 0] = 1.0
        self.sobely.weight.data[0, 0, 0, 1] = 2.0
        self.sobely.weight.data[0, 0, 0, 2] = 1.0

        self.sobely.weight.data[0, 0, 1, 0] = 0.0
        self.sobely.weight.data[0, 0, 1, 1] = 0.0
        self.sobely.weight.data[0, 0, 1, 2] = 0.0

        self.sobely.weight.data[0, 0, 2, 0] = -1.0
        self.sobely.weight.data[0, 0, 2, 1] = -2.0
        self.sobely.weight.data[0, 0, 2, 2] = -1.0
        self.sobely.weight.requires_grad = False
        self.conv_after_sobely = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)
    def forward(self, x):
        y = self.sobely(x)
        y = self.conv_after_sobely(y)
        return y

class Lapl(nn.Module):
    def __init__(self, inp_planes, out_planes):
        super(Lapl, self).__init__()
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.lapl = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.lapl.weight.data[0, 0, 0, 0] = 0.0
        self.lapl.weight.data[0, 0, 1, 0] = 1.0
        self.lapl.weight.data[0, 0, 2, 0] = 0.0

        self.lapl.weight.data[0, 0, 0, 1] = 1.0
        self.lapl.weight.data[0, 0, 1, 1] = -4.0
        self.lapl.weight.data[0, 0, 2, 1] = 1.0

        self.lapl.weight.data[0, 0, 0, 2] = 0.0
        self.lapl.weight.data[0, 0, 1, 2] = 1.0
        self.lapl.weight.data[0, 0, 2, 2] = 0.0
        self.lapl.weight.requires_grad = False
        self.conv_after_lapl = nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=3, padding=1)

    def forward(self, x):
        y = self.lapl(x)
        y = self.conv_after_lapl(y)
        return y

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape 

        xz = self.in_proj(x) # B, H, W, 2C
        x, z = xz.chunk(2, dim=-1) # B, H, W, C, and B, H, W, C

        x = x.permute(0, 3, 1, 2).contiguous() # B, C, H, W
        x = self.act(self.conv2d(x)) # B, C, H, W
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class S_SSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,expand=expand,dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path)
        self.scale_mapping = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(True),
            nn.Linear(16, 1)
        )

    def forward(self, input, x_size, sr_scale):
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  
        x = self.ln_1(input)
        sr_scale = sr_scale.clone()
        mapping_weights = self.scale_mapping(sr_scale).view(B,1,1,1)
        x = input*mapping_weights + self.drop_path(self.self_attention(x))
        x = x.view(B, -1, C).contiguous()
        return x

class SSSB(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
    """
    def __init__(self,
                 dim,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 is_light_sr=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.mlp_ratio=mlp_ratio

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(S_SSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,is_light_sr=is_light_sr))

        # patch merging layer
        self.downsample = None

    def forward(self, x, x_size, sr_scale):
        for blk in self.blocks:
            x = blk(x, x_size, sr_scale)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, depth={self.depth}'

class Scale_Specific_State_Space_Group(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 d_state=16,
                 mlp_ratio=4.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 resi_connection='1conv'):
        super(Scale_Specific_State_Space_Group, self).__init__()

        self.dim = dim

        self.residual_group = SSSB(
            dim=dim,
            depth=depth,
            d_state = d_state,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample)

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

        self.patch_embed = PatchEmbed(embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(embed_dim=dim)

        self.routing = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(True),
            nn.Linear(32, 4),
            nn.Softmax(1)
        )

        weight_pool = []
        for i in range(4):
            weight_pool.append(nn.Parameter(torch.Tensor(dim, dim)))
            nn.init.kaiming_uniform_(weight_pool[i], math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

    def forward(self, x, x_size, sr_scale):
        B,L,C = x.size()
        device = x.device
        
        sr_scale = sr_scale / torch.ones(1, 1).to(device)
        sr_scale = sr_scale.clone()
        routing_weights = self.routing(sr_scale).view(B, 4, 1)

        fused_weight = (self.weight_pool.view(4, -1) * routing_weights).sum(1)
        fused_weight = fused_weight.view(-1,C, C)
        out_lst = []
        for i in range(B):
            out = F.linear(x[i].unsqueeze(0), fused_weight[i], None)
            out_lst.append(out)
        out = torch.cat(out_lst, 0)
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, sr_scale), x_size))) + out


class PatchEmbed(nn.Module):
    def __init__(self, embed_dim=96, norm_layer=None):
        super().__init__()
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

class Scale_Specific_Encoder(nn.Module): 
    def __init__(self,
                 in_chans=1,
                 embed_dim=64,
                 depths=(3, 3, 3, 3),
                 drop_rate=0.,
                 d_state = 16,
                 mlp_ratio=2.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 upscale=2,
                 img_range=1.,
                 resi_connection='1conv',
                 **kwargs):
        super(Scale_Specific_Encoder, self).__init__()
        num_in_ch = in_chans
        num_feat = 64
        self.img_range = img_range

        self.upscale = upscale
        self.mlp_ratio=mlp_ratio
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
       
        # return 2D feature map from 1D token sequence
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers): # 6-layer
            layer = Scale_Specific_State_Space_Group(
                dim=embed_dim,
                depth=depths[i_layer],
                d_state = d_state,
                mlp_ratio=self.mlp_ratio,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],  
                norm_layer=norm_layer,
                downsample=None,
                resi_connection=resi_connection
            )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))
        
        self.sbx = Sobelx(num_in_ch, embed_dim)
        self.sby = Sobely(num_in_ch, embed_dim)
        self.lpl = Lapl(num_in_ch, embed_dim)

        self.conv_after_add = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.conv_before_upsample = nn.Sequential(
            nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, scale):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x) # N,L,C

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size, scale)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x, scale):
        fea = self.conv_first(x)
        fea = self.conv_after_body(self.forward_features(fea, scale)) + fea

        y = self.sbx(x) + self.sby(x) + self.lpl(x)
        y = self.conv_after_add(y)

        fea = self.conv_before_upsample(fea+y)
        return fea

@register('Scale-Specific-Encoder-baseline')
def make_scale_specific_encoder_baseline(no_upsampling=True):
    return Scale_Specific_Encoder()

