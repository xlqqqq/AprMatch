import torch
import torch.nn.functional as F
from einops import rearrange


def apr_mix(img_source, img_target, prob_source, prob_target, patch_size=64, top_k=4):

    B, C, H, W = img_source.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size

    if pad_h > 0 or pad_w > 0:

        img_source  = F.pad(img_source,  (0, pad_w, 0, pad_h))
        img_target  = F.pad(img_target,  (0, pad_w, 0, pad_h))
        prob_source = F.pad(prob_source, (0, pad_w, 0, pad_h))
        prob_target = F.pad(prob_target, (0, pad_w, 0, pad_h))


    conf_source = torch.max(prob_source, dim=1)[0]
    conf_target = torch.max(prob_target, dim=1)[0]

    patches_conf_s = rearrange(conf_source, 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                               p1=patch_size, p2=patch_size)
    patches_conf_t = rearrange(conf_target, 'b (h p1) (w p2) -> b (h w) (p1 p2)',
                               p1=patch_size, p2=patch_size)

    mean_conf_s = torch.mean(patches_conf_s, dim=2)
    mean_conf_t = torch.mean(patches_conf_t, dim=2)

    img_patches_s = rearrange(img_source, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                              p1=patch_size, p2=patch_size)
    img_patches_apr = rearrange(img_target, 'b c (h p1) (w p2) -> b (h w) (c p1 p2)',
                                p1=patch_size, p2=patch_size).clone()


    actual_k = min(top_k, mean_conf_t.shape[1])

    for b in range(B):
        _, low_conf_indices = torch.topk(mean_conf_t[b], k=actual_k, largest=False)
        for idx in low_conf_indices:
            # 在该位置足够可靠时才替换，避免引入噪声
            if mean_conf_s[b, idx] > 0.90:
                img_patches_apr[b, idx] = img_patches_s[b, idx]


    h_num = (H + pad_h) // patch_size
    w_num = (W + pad_w) // patch_size

    img_apr = rearrange(img_patches_apr, 'b (h w) (c p1 p2) -> b c (h p1) (w p2)',
                        h=h_num, w=w_num, p1=patch_size, p2=patch_size, c=C)

    if pad_h > 0 or pad_w > 0:
        img_apr = img_apr[:, :, :H, :W]

    return img_apr
