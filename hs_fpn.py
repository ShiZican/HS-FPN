#------------------------------------------------------------------#
# Code Structure of HS-FPN (https://arxiv.org/abs/2412.10116)
# HS-FPN
# ├── HFP (High Frequency Perception Module)
# │   ├── DctSpatialInteraction (Spatial Path of HFP)
# │   └── DctChannelInteraction (Channel Path of HFP)
# └── SDP&SDP_Large (Spatial Dependency Perception Module
#-----------------------------------------------------------------#

import math
import torch
import numpy as np
import torch.nn as nn
import torch_dct as DCT
import torch.nn.functional as F
from einops import rearrange
from mmcv.cnn import ConvModule
from mmdet.models.builder import NECKS
from mmcv.runner import BaseModule, auto_fp16

__all__ =['HS-FPN']

#------------------------------------------------------------------#
# Spatial Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctSpatialInteraction(BaseModule):
    def __init__(self,
                in_channels,
                ratio,
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(DctSpatialInteraction, self).__init__(init_cfg)
        self.ratio = ratio
        self.isdct = isdct # true when in p1&p2 # false when in p3&p4
        if not self.isdct:
            self.spatial1x1 = nn.Sequential(
            *[ConvModule(in_channels, 1, kernel_size=1, bias=False)]
        )

    def forward(self, x):
        _, _, h0, w0 = x.size()
        if not self.isdct:
            return x * torch.sigmoid(self.spatial1x1(x))
        idct = DCT.dct_2d(x, norm='ortho') 
        weight = self._compute_weight(h0, w0, self.ratio).to(x.device)
        weight = weight.view(1, h0, w0).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') # generate spatial mask
        return x * dct_

    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight


#------------------------------------------------------------------#
# Channel Path of HFP
# Only p1&p2 use dct to extract high_frequency response
#------------------------------------------------------------------#
class DctChannelInteraction(BaseModule):
    def __init__(self,
                in_channels, 
                patch,
                ratio,
                isdct=True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')
                ):
        super(DctChannelInteraction, self).__init__(init_cfg)
        self.in_channels = in_channels
        self.h = patch[0]
        self.w = patch[1]
        self.ratio = ratio
        self.isdct = isdct
        self.channel1x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.channel2x1 = nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=1, groups=32, bias=False)],
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        n, c, h, w = x.size()
        if not self.isdct: # true when in p1&p2 # false when in p3&p4
            amaxp = F.adaptive_max_pool2d(x,  output_size=(1, 1))
            aavgp = F.adaptive_avg_pool2d(x,  output_size=(1, 1))
            channel = self.channel1x1(self.relu(amaxp)) + self.channel1x1(self.relu(aavgp)) # 2025 03 15 szc 
            return x * torch.sigmoid(self.channel2x1(channel))

        idct = DCT.dct_2d(x, norm='ortho')
        weight = self._compute_weight(h, w, self.ratio).to(x.device)
        weight = weight.view(1, h, w).expand_as(idct)             
        dct = idct * weight # filter out low-frequency features 
        dct_ = DCT.idct_2d(dct, norm='ortho') 

        amaxp = F.adaptive_max_pool2d(dct_,  output_size=(self.h, self.w))
        aavgp = F.adaptive_avg_pool2d(dct_,  output_size=(self.h, self.w))       
        amaxp = torch.sum(self.relu(amaxp), dim=[2,3]).view(n, c, 1, 1)
        aavgp = torch.sum(self.relu(aavgp), dim=[2,3]).view(n, c, 1, 1)

        # channel = torch.cat([self.channel1x1(aavgp), self.channel1x1(amaxp)], dim = 1) # TODO: The values of aavgp and amaxp appear to be on different scales. Add is a better choice instead of concate.
        channel = self.channel1x1(amaxp) + self.channel1x1(aavgp) # 2025 03 15 szc 
        return x * torch.sigmoid(self.channel2x1(channel))
        
    def _compute_weight(self, h, w, ratio):
        h0 = int(h * ratio[0])
        w0 = int(w * ratio[1])
        weight = torch.ones((h, w), requires_grad=False)
        weight[:h0, :w0] = 0
        return weight  


#------------------------------------------------------------------#
# High Frequency Perception Module HFP
#------------------------------------------------------------------#
class HFP(BaseModule):
    def __init__(self, 
                in_channels,
                ratio,
                patch = (8,8),
                isdct = True,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HFP, self).__init__(init_cfg)
        self.spatial = DctSpatialInteraction(in_channels, ratio=ratio, isdct = isdct) 
        self.channel = DctChannelInteraction(in_channels, patch=patch, ratio=ratio, isdct = isdct)
        self.out =  nn.Sequential(
            *[ConvModule(in_channels, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, in_channels)]
            )
    def forward(self, x):
        spatial = self.spatial(x) # output of spatial path
        channel = self.channel(x) # output of channel path
        return self.out(spatial + channel)


#------------------------------------------------------------------#
# Spatial Dependency Perception Module SDP
#------------------------------------------------------------------#
class SDP(BaseModule):
    def __init__(self,
                dim=256,
                inter_dim=None,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP, self).__init__(init_cfg)
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[ConvModule(dim, self.inter_dim, 1, padding=0, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        return output + x_low

#------------------------------------------------------------------#
# Improved Version of Spatial Dependency Perception Module SDP
# 2025 03 15 szc 
#------------------------------------------------------------------#
class SDP_Improved(BaseModule):
    def __init__(self,
                dim=256,
                inter_dim=None,
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(SDP_Improved, self).__init__(init_cfg)
        self.inter_dim=inter_dim
        if self.inter_dim == None:
            self.inter_dim = dim
        self.conv_q = nn.Sequential(*[ConvModule(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv_k = nn.Sequential(*[ConvModule(dim, self.inter_dim, 3, padding=1, bias=False), nn.GroupNorm(32,self.inter_dim)])
        self.conv = nn.Sequential(*[ConvModule(self.inter_dim, dim, 3, padding=1, bias=False), nn.GroupNorm(32, dim)])
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_low, x_high, patch_size):
        b_, _, h_, w_ = x_low.size()
        q = rearrange(self.conv_q(x_low), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        q = q.transpose(1,2) # 1,4096,128
        k = rearrange(self.conv_k(x_high), 'b c (h p1) (w p2) -> (b h w) c (p1 p2)', p1=patch_size[0], p2=patch_size[1])
        attn = torch.matmul(q, k) # 1, 4096, 1024
        attn = attn / np.power(self.inter_dim, 0.5)
        attn = self.softmax(attn)
        v = k.transpose(1,2)# 1, 1024, 128
        output = torch.matmul(attn,v)# 1, 4096, 128
        output = rearrange(output.transpose(1, 2).contiguous(), '(b h w) c (p1 p2) -> b c (h p1) (w p2)', p1=patch_size[0], p2=patch_size[1], h=h_//patch_size[0], w=w_//patch_size[1])
        output = self.conv(output + x_low)
        return output


#------------------------------------------------------------------#
# HS_FPN
#------------------------------------------------------------------#
# @NECKS.register_module()
class HS_FPN(BaseModule):
    def __init__(self,
                in_channels,
                out_channels,
                num_outs,
                ratio = (0.25, 0.25),
                start_level=0,
                end_level=-1,
                add_extra_convs=False,
                relu_before_extra_convs=False,
                no_norm_on_lateral=False,
                conv_cfg=None,
                norm_cfg=None,
                act_cfg=None,
                upsample_cfg=dict(mode='nearest'),
                init_cfg=dict(
                    type='Xavier', layer='Conv2d', distribution='uniform')):
        super(HS_FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
        def interpolate(input):
            up_mode = 'nearest'
            return F.interpolate(input, scale_factor=2, mode='nearest', align_corners=False if up_mode == 'bilinear' else None)
        self.fpn_upsample = interpolate 
        self.SelfAttn_p4 = HFP(out_channels, ratio=None, isdct=False)
        self.SelfAttn_p3 = HFP(out_channels, ratio=None, isdct=False)
        self.SelfAttn_p2 = HFP(out_channels, ratio=ratio, patch=(8, 8), isdct=True)
        self.SelfAttn_p1 = HFP(out_channels, ratio=ratio, patch=(16, 16), isdct=True)

        self.CrossAtten_p4_p3 = SDP(dim=out_channels)
        self.CrossAtten_p3_p2 = SDP(dim=out_channels)
        self.CrossAtten_p2_p1 = SDP(dim=out_channels)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        _, _, h, w = laterals[3].size()
        laterals[3] = self.SelfAttn_p4(laterals[3])
        laterals[2] = self.CrossAtten_p4_p3(self.SelfAttn_p3(laterals[2]), self.fpn_upsample(laterals[3]), [h, w])
        laterals[1] = self.CrossAtten_p3_p2(self.SelfAttn_p2(laterals[1]), self.fpn_upsample(laterals[2]), [h, w])
        laterals[0] = self.CrossAtten_p2_p1(self.SelfAttn_p1(laterals[0]), self.fpn_upsample(laterals[1]), [h, w])

        used_backbone_levels = len(laterals)  
        
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            
        # # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)