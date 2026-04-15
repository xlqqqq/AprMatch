import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbone import mit


class ECALayer(nn.Module):
    """Efficient Channel Attention (lightweight)."""
    def __init__(self, channels, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)


class SpatialGate(nn.Module):
    """Light spatial attention gate."""
    def __init__(self, in_ch, mid_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, 1, kernel_size=1, bias=True)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


class ProgressiveFusionGate(nn.Module):
    def __init__(self, dim, use_eca=True, use_spatial=True):
        super().__init__()
        self.use_eca = use_eca
        self.use_spatial = use_spatial

        if use_eca:
            self.eca = ECALayer(dim, k_size=3)
        if use_spatial:
            self.spatial = SpatialGate(in_ch=dim * 2, mid_ch=max(dim // 8, 16))

        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, up_feat, skip_feat, gate_scale=1.0):
        if self.use_eca:
            skip_feat = self.eca(skip_feat)
        if self.use_spatial:
            g = self.spatial(torch.cat([up_feat, skip_feat], dim=1))
            fused = up_feat + (gate_scale * g) * skip_feat
        else:
            fused = up_feat + skip_feat
        return self.proj(fused)


class LiteASPP(nn.Module):
    """Lightweight ASPP context module for thin/long structures (roads)."""
    def __init__(self, in_ch, mid_ch=64, rates=(1, 6, 12, 18)):
        super().__init__()
        mid_ch = int(mid_ch)
        branches = []
        for r in rates:
            if r == 1:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                ))
            else:
                branches.append(nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=r, dilation=r, bias=False),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                ))
        self.branches = nn.ModuleList(branches)
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_ch * len(rates), in_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.fuse(torch.cat([b(x) for b in self.branches], dim=1))


class SegFormer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.nclass = cfg['nclass']
        self.backbone_name = cfg['backbone']

        all_backbones = {**mit.__dict__}
        if self.backbone_name not in all_backbones:
            raise NotImplementedError(f"Backbone '{self.backbone_name}' is not implemented.")

        self.backbone = all_backbones[self.backbone_name](pretrained=True)
        encoder_channels = self.backbone.embed_dims

        decoder_dim = cfg.get('decoder_dim', 256)
        self.decoder_dim = decoder_dim

        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_channels[i], decoder_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(decoder_dim),
                nn.ReLU(inplace=True)
            ) for i in range(4)
        ])

        self.num_prog_gates = int(cfg.get('num_prog_gates', 3))
        self.prog_gates = nn.ModuleList([
            ProgressiveFusionGate(
                dim=decoder_dim,
                use_eca=bool(cfg.get('pua_use_eca', True)),
                use_spatial=bool(cfg.get('pua_use_spatial', True)),
            ) for _ in range(self.num_prog_gates)
        ])

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True)
        )

        self.seg_head = nn.Conv2d(decoder_dim, self.nclass, kernel_size=1)

        bdy_mid = max(decoder_dim // 4, 16)
        self.boundary_trunk = nn.Sequential(
            nn.Conv2d(decoder_dim, bdy_mid, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(bdy_mid),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )
        self.boundary_pred = nn.Conv2d(bdy_mid, 1, kernel_size=1)
        self.boundary_gate = nn.Conv2d(bdy_mid, decoder_dim, kernel_size=1)

        self.gate_scale = float(cfg.get('gate_scale', 1.0))
        self.pua_gate_scale = float(cfg.get('pua_gate_scale', 1.0))

        aspp_mid = int(cfg.get('aspp_mid', max(decoder_dim // 4, 64)))
        aspp_rates = cfg.get('aspp_rates', (1, 6, 12, 18))
        self.use_aspp = bool(cfg.get('use_aspp', True))
        if self.use_aspp:
            self.aspp = LiteASPP(decoder_dim, mid_ch=aspp_mid, rates=aspp_rates)

        self.edge_sharpen = float(cfg.get('edge_sharpen', 0.2))

        # Pre-defined dropout for feature perturbation (need_fp)
        self.feat_drop = nn.Dropout2d(0.5)

    def _progressive_decode(self, features, pua_gate_scale=None):
        if pua_gate_scale is None:
            pua_gate_scale = self.pua_gate_scale

        c1 = self.proj[0](features[0])
        c2 = self.proj[1](features[1])
        c3 = self.proj[2](features[2])
        c4 = self.proj[3](features[3])

        x = c4
        if self.num_prog_gates >= 1:
            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False)
            x = self.prog_gates[0](x, c3, gate_scale=pua_gate_scale)
        else:
            x = F.interpolate(x, size=c3.shape[-2:], mode='bilinear', align_corners=False) + c3

        if self.num_prog_gates >= 2:
            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False)
            x = self.prog_gates[1](x, c2, gate_scale=pua_gate_scale)
        else:
            x = F.interpolate(x, size=c2.shape[-2:], mode='bilinear', align_corners=False) + c2

        if self.num_prog_gates >= 3:
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False)
            x = self.prog_gates[2](x, c1, gate_scale=pua_gate_scale)
        else:
            x = F.interpolate(x, size=c1.shape[-2:], mode='bilinear', align_corners=False) + c1

        return self.fuse_conv(x)

    def _boundary_refine(self, fused, gate_scale=None):
        if gate_scale is None:
            gate_scale = self.gate_scale

        bdy_feat = self.boundary_trunk(fused)
        bdy_logits = self.boundary_pred(bdy_feat)
        gate = torch.sigmoid(self.boundary_gate(bdy_feat))
        fused_refined = fused * (1.0 + gate_scale * gate)

        if self.edge_sharpen > 0:
            bdy_prob = torch.sigmoid(bdy_logits)
            fused_refined = fused_refined + self.edge_sharpen * bdy_prob * fused_refined

        return fused_refined, bdy_logits

    def forward(self, x, need_fp=False, gate_scale=None, pua_gate_scale=None):
        h, w = x.shape[-2:]
        features = self.backbone.base_forward(x)

        if need_fp:
            features_fp = [self.feat_drop(feat) for feat in features]
            doubled_features = [
                torch.cat([clean, perturbed], dim=0)
                for clean, perturbed in zip(features, features_fp)
            ]

            fused = self._progressive_decode(doubled_features, pua_gate_scale=pua_gate_scale)
            if self.use_aspp:
                fused = fused + self.aspp(fused)

            fused_refined, bdy_outs = self._boundary_refine(fused, gate_scale=gate_scale)
            seg_outs = self.seg_head(fused_refined)

            seg_outs = F.interpolate(seg_outs, size=(h, w), mode='bilinear', align_corners=False)
            bdy_outs = F.interpolate(bdy_outs, size=(h, w), mode='bilinear', align_corners=False)

            seg_out, seg_out_fp = seg_outs.chunk(2)
            bdy_out, bdy_out_fp = bdy_outs.chunk(2)
            return (seg_out, bdy_out), (seg_out_fp, bdy_out_fp)

        fused = self._progressive_decode(features, pua_gate_scale=pua_gate_scale)
        if self.use_aspp:
            fused = fused + self.aspp(fused)

        fused_refined, bdy_out = self._boundary_refine(fused, gate_scale=gate_scale)
        seg_out = self.seg_head(fused_refined)

        seg_out = F.interpolate(seg_out, size=(h, w), mode='bilinear', align_corners=False)
        bdy_out = F.interpolate(bdy_out, size=(h, w), mode='bilinear', align_corners=False)

        if self.training:
            return seg_out, bdy_out
        else:
            return seg_out
