
from config.conf import cfg
from typing import Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.misc import is_main_process


def hand_loss(pre_hand, targets):
    pred_mano_results, gt_mano_results, preds_joints_img = pre_hand

    loss = {}
    loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'],
                                                            gt_mano_results['verts3d'])
    loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'],
                                                              gt_mano_results['joints3d'])
    loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'],
                                                          gt_mano_results['mano_pose'])
    loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'],
                                                            gt_mano_results['mano_shape'])
    loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints_img'])

    return loss

class SILog(nn.Module):
    def __init__(self, weight: float):
        super(SILog, self).__init__()
        self.name: str = "SILog"
        self.weight = weight
        self.eps: float = 1e-6

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        interpolate: bool = True,
    ) -> torch.Tensor:
        if interpolate:
            input = F.interpolate(
                input, target.shape[-2:], mode="bilinear", align_corners=True
            )
        if mask is not None:
            input = input[mask]
            target = target[mask]

        log_error = torch.log(input + self.eps) - torch.log(target + self.eps)
        mean_sq_log_error = torch.pow(torch.mean(log_error), 2.0)

        scale_inv = torch.var(log_error)
        Dg = scale_inv + 0.15 * mean_sq_log_error
        return torch.sqrt(Dg + self.eps)

    @classmethod
    def build(cls, config):
        return cls(weight=config["training"]["loss"]["weight"])

def normalize_normals(norms):
    min_kappa = 0.01
    norm_x, norm_y, norm_z, kappa = torch.split(norms, 1, dim=1)
    norm = torch.sqrt(norm_x**2.0 + norm_y**2.0 + norm_z**2.0 + 1e-6)
    kappa = F.elu(kappa) + 1.0 + min_kappa
    norms = torch.cat([norm_x / norm, norm_y / norm, norm_z / norm, kappa], dim=1)
    return norms


def idisc_loss(out, gt, original_shape=256, weight = 10.0, mask = None):
    loss = SILog(weight)
    out_lst = []

    if out.shape[1] == 1:
        out = F.interpolate(
            torch.exp(out),
            size=out.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
    else:
        out = normalize_normals(
            F.interpolate(
                out,
                size=out.shape[-2:],
                mode="bilinear",
                align_corners=True,
            )
        )
    out_lst.append(out)


    out = F.interpolate(
        torch.mean(torch.stack(out_lst, dim=0), dim=0),
        original_shape,
        # Legacy code for reproducibility for normals...
        mode="bilinear" if out.shape[1] == 1 else "bicubic",
        align_corners=True,
    )

    if gt is not None:
        SiLog = weight * loss(out, target=gt, mask=mask, interpolate=True)

    return SiLog





