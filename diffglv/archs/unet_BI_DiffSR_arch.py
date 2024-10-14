import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

from basicsr.utils.registry import ARCH_REGISTRY



def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# --------------------------------------------- BI Basic Units: START -----------------------------------------------------------------
class RPReLU(nn.Module):
    def __init__(self, inplanes):
        super(RPReLU, self).__init__()
        self.pr_bias0 = LearnableBias(inplanes)
        self.pr_prelu = nn.PReLU(inplanes)
        self.pr_bias1 = LearnableBias(inplanes)

    def forward(self, x):
        x = self.pr_bias1(self.pr_prelu(self.pr_bias0(x)))
        return x

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out

class HardBinaryConv(nn.Conv2d):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1,groups=1,bias=True):
        super(HardBinaryConv, self).__init__(
            in_chn,
            out_chn,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias
        )

    def forward(self, x):
        real_weights = self.weight
        scaling_factor = torch.mean(torch.mean(torch.mean(abs(real_weights),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True)
        scaling_factor = scaling_factor.detach()
        binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights,self.bias, stride=self.stride, padding=self.padding)
        return y

class BIConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, groups=1, dynamic_group=5):
        super(BIConv, self).__init__()
        self.TaR = nn.ModuleDict({
            f'dynamic_move_{i}': LearnableBias(in_channels) for i in range(dynamic_group)
        })

        self.binary_activation = BinaryActivation()
        self.binary_conv = HardBinaryConv(in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size//2),
        bias=bias)
        self.TaA = nn.ModuleDict({
            f'dynamic_relu_{i}': RPReLU(out_channels) for i in range(dynamic_group)
        })

    def forward(self, x, t):
        out = self.TaR[f'dynamic_move_{t}'](x)
        out = self.binary_activation(out)
        out = self.binary_conv(out)
        out = self.TaA[f'dynamic_relu_{t}'](out)
        out = out + x
        return out
# --------------------------------------------- BI Basic Units: END -----------------------------------------------------------------

# --------------------------------------------- FP Module: START --------------------------------------------------------------------
# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=timestep_level.dtype,
                            device=timestep_level.device) / count
        encoding = timestep_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        )

        assert dim == dim_out, f"Error: input ({dim}) and output ({dim_out}) channel dimension."

        if dim == dim_out:
            self.conv = nn.Conv2d(dim, dim_out, 3, padding=1)

    def forward(self, x):
        return self.conv(self.block(x))

class Block_F(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input

class CP_Up_FP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.biconv1 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.biconv2 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        '''
        input: b,c,h,w
        output: b,c/2,h*2,w*2
        '''
        out1 = self.biconv1(x)
        out2 = self.biconv2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.up(out)
        return out

class CP_Down_FP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.biconv1 = nn.Conv2d(dim//2, dim//2, 3, padding=1)
        self.biconv2 = nn.Conv2d(dim//2, dim//2, 3, padding=1)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x):
        '''
        input: b,c,h,w
        output: b,2c,h/2,w/2
        '''
        b,c,h,w = x.shape
        out1 = self.biconv1(x[:,:c//2,:,:])
        out2 = self.biconv2(x[:,c//2:,:,:])
        out = out1 + out2
        out = self.down(out)
        return out

class CS_Fusion_FP(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=1):
        super(CS_Fusion_FP, self).__init__()

        assert in_channels // 2 == out_channels, f"Error: input ({in_channels}) and output ({out_channels}) channel dimension."

        self.biconv_1 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.biconv_2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding, groups=groups)

    def forward(self, x):
        '''
        x: b,c,h,w
        out: b,c/2,h,w
        '''
        b,c,h,w = x.shape
        in_1 = x[:,:c//2,:,:]
        in_2 = x[:,c//2:,:,:]

        fu_1 = torch.cat((in_1[:, 1::2, :, :], in_2[:, 0::2, :, :]), dim=1)
        fu_2 = torch.cat((in_1[:, 0::2, :, :], in_2[:, 1::2, :, :]), dim=1)

        out_1 = self.biconv_1(fu_1)
        out_2 = self.biconv_2(fu_2)
        
        out = out_1 + out_2
        return out

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, timestep_level_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.timestep_func = nn.Sequential(
            Swish(),
            nn.Linear(timestep_level_emb_dim, dim_out)
        )

        assert dim == dim_out, f"Error: input ({dim}) and output ({dim_out}) channel dimension."

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        if dim == dim_out:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        t_emb = self.timestep_func(time_emb).type(h.dtype)
        h = h + t_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, timestep_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, timestep_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x
# -------------------------------------------------- FP Module: END ----------------------------------------------------------------------

# -------------------------------------------------- BI Module: START --------------------------------------------------------------------
class CP_Up(nn.Module):
    def __init__(self, dim, dynamic_group=5):
        super().__init__()
        self.biconv1 = BIConv(dim, dim, 3, 1, 1, dynamic_group=dynamic_group)
        self.biconv2 = BIConv(dim, dim, 3, 1, 1, dynamic_group=dynamic_group)
        self.up = nn.PixelShuffle(2)

    def forward(self, x, t):
        '''
        input: b,c,h,w
        output: b,c/2,h*2,w*2
        '''
        out1 = self.biconv1(x, t)
        out2 = self.biconv2(x, t)
        out = torch.cat([out1, out2], dim=1)
        out = self.up(out)
        return out

class CP_Down(nn.Module):
    def __init__(self, dim, dynamic_group=5):
        super().__init__()
        self.biconv1 = BIConv(dim//2, dim//2, 3, padding=1, dynamic_group=dynamic_group)
        self.biconv2 = BIConv(dim//2, dim//2, 3, padding=1, dynamic_group=dynamic_group)
        self.down = nn.PixelUnshuffle(2)

    def forward(self, x, t):
        '''
        input: b,c,h,w
        output: b,2c,h/2,w/2
        '''
        b,c,h,w = x.shape
        out1 = self.biconv1(x[:,:c//2,:,:], t)
        out2 = self.biconv2(x[:,c//2:,:,:], t)
        out = out1 + out2
        out = self.down(out)
        return out

class CS_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dynamic_group=5):
        super(CS_Fusion, self).__init__()

        assert in_channels // 2 == out_channels, f"Error: input ({in_channels}) and output ({out_channels}) channel dimension."

        self.biconv_1 = BIConv(out_channels, out_channels, kernel_size, stride, padding, bias, groups, dynamic_group=dynamic_group)
        self.biconv_2 = BIConv(out_channels, out_channels, kernel_size, stride, padding, bias, groups, dynamic_group=dynamic_group)

    def forward(self, x, t):
        '''
        x: b,c,h,w
        out: b,c/2,h,w
        '''
        b,c,h,w = x.shape
        in_1 = x[:,:c//2,:,:]
        in_2 = x[:,c//2:,:,:]

        fu_1 = torch.cat((in_1[:, 1::2, :, :], in_2[:, 0::2, :, :]), dim=1)
        fu_2 = torch.cat((in_1[:, 0::2, :, :], in_2[:, 1::2, :, :]), dim=1)

        out_1 = self.biconv_1(fu_1, t)
        out_2 = self.biconv_2(fu_2, t)
        
        out = out_1 + out_2
        return out

class BI_Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0, dynamic_group=5):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
        )

        assert dim == dim_out, f"Error: input ({dim}) and output ({dim_out}) channel dimension."

        if dim == dim_out:
            self.conv = BIConv(dim, dim_out, 3, padding=1, dynamic_group=dynamic_group)

    def forward(self, x, t):
        return self.conv(self.block(x), t)

class BI_ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, timestep_level_emb_dim=None, dropout=0, norm_groups=32, dynamic_group=5):
        super().__init__()
        self.timestep_func = nn.Sequential(
            Swish(),
            nn.Linear(timestep_level_emb_dim, dim_out)
        )

        assert dim == dim_out, f"Error: input ({dim}) and output ({dim_out}) channel dimension."

        self.block1 = BI_Block(dim, dim_out, groups=norm_groups, dynamic_group=dynamic_group)
        self.block2 = BI_Block(dim_out, dim_out, groups=norm_groups, dropout=dropout, dynamic_group=dynamic_group)
        if dim == dim_out:
            self.res_conv = nn.Identity()

    def forward(self, x, time_emb, t):
        b, c, h, w = x.shape
        h = self.block1(x, t)
        t_emb = self.timestep_func(time_emb).type(h.dtype)
        h = h + t_emb[:, :, None, None]
        h = self.block2(h, t)
        return h + self.res_conv(x)

class BI_ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, timestep_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False, dynamic_group=5):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = BI_ResnetBlock(
            dim, dim_out, timestep_level_emb_dim, norm_groups=norm_groups, dropout=dropout, dynamic_group=dynamic_group)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb, t):
        x = self.res_block(x, time_emb, t)
        if(self.with_attn):
            x = self.attn(x)
        return x
# -------------------------------------------------- BI Module: END --------------------------------------------------------------------

# ----------------------------------------------- BI-DiffSR UNet: START ----------------------------------------------------------------
@ARCH_REGISTRY.register()
class BIDiffSRUNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=(8),
        res_blocks=3,
        dropout=0,
        image_size=128,
        fp_res=(0),
        total_step=2000,
        dynamic_group=5
    ):
        super(BIDiffSRUNet, self).__init__()

        self.in_channel = in_channel
        self.total_step = total_step
        self.dynamic_group = dynamic_group

        timestep_level_channel = inner_channel
        self.timestep_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(BI_ResnetBlocWithAttn(
                        pre_channel, channel_mult, timestep_level_emb_dim=timestep_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn, dynamic_group=dynamic_group))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(CP_Down(pre_channel, dynamic_group=dynamic_group))
                now_res = now_res//2
                pre_channel = pre_channel*2
                feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            BI_ResnetBlocWithAttn(pre_channel, pre_channel, timestep_level_emb_dim=timestep_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True, dynamic_group=dynamic_group),
            BI_ResnetBlocWithAttn(pre_channel, pre_channel, timestep_level_emb_dim=timestep_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False, dynamic_group=dynamic_group)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            use_fp= (now_res in fp_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                if use_fp:
                    ups.append(CS_Fusion_FP(pre_channel+feat_channels.pop(), channel_mult, kernel_size=1, stride=1, padding=0, bias=False, groups=1))
                    ups.append(ResnetBlocWithAttn(
                        channel_mult, channel_mult, timestep_level_emb_dim=timestep_level_channel, norm_groups=norm_groups,
                            dropout=dropout, with_attn=use_attn))
                else:
                    ups.append(CS_Fusion(pre_channel+feat_channels.pop(), channel_mult, kernel_size=1, stride=1, padding=0, bias=False, groups=1, dynamic_group=dynamic_group))
                    ups.append(BI_ResnetBlocWithAttn(
                        channel_mult, channel_mult, timestep_level_emb_dim=timestep_level_channel, norm_groups=norm_groups,
                            dropout=dropout, with_attn=use_attn, dynamic_group=dynamic_group))
                pre_channel = channel_mult
            if not is_last:
                ups.append(CP_Up(pre_channel, dynamic_group=dynamic_group))
                now_res = now_res*2
                pre_channel = pre_channel//2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block_F(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, c, time):
        index_dynamic = int(time[0][0] * self.dynamic_group / self.total_step)
        index_dynamic = max(0, min(index_dynamic, self.dynamic_group - 1))

        time = time.squeeze(1) # consistent with the original code

        if self.in_channel != 3:
            x = torch.cat([c, x], dim=1)
        t = self.timestep_level_mlp(time) if exists(
            self.timestep_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, BI_ResnetBlocWithAttn):
                x = layer(x, t, index_dynamic)
            elif isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            elif isinstance(layer, CP_Down):
                x = layer(x, index_dynamic)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, BI_ResnetBlocWithAttn):
                x = layer(x, t, index_dynamic)
            else:
                x = layer(x)
        
        for layer in self.ups:
            if isinstance(layer, CS_Fusion):
                x = layer(torch.cat((x, feats.pop()), dim=1), index_dynamic)
            elif isinstance(layer, CS_Fusion_FP):
                x = layer(torch.cat((x, feats.pop()), dim=1))
            elif isinstance(layer, BI_ResnetBlocWithAttn):
                x = layer(x, t, index_dynamic)
            elif isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            elif isinstance(layer, CP_Up):
                x = layer(x, index_dynamic)
            else:
                x = layer(x)

        return self.final_conv(x)


if __name__ == '__main__':
    model = BIDiffSRUNet(
            in_channel = 6,
            out_channel = 3,
            inner_channel = 64,
            norm_groups = 16,
            channel_mults = [1, 2, 4, 8],
            attn_res = [],
            res_blocks = 2,
            dropout = 0.2,
            image_size = 256,
            fp_res= [256, 128],
            dynamic_group=5
    )
    print(model)

    x = torch.randn((2, 3, 128, 128))
    c = torch.randn((2, 3, 128, 128))
    timesteps = torch.randint(0, 10, (2,)).long().unsqueeze(1)
    x = model(x, c, timesteps)
    print(x.shape)
    print(sum(map(lambda x: x.numel(), model.parameters())))