import argparse
import logging
import os
import pprint
import random
import numpy as np

import torch
from torch import nn
import torch.backends.cudnn as cudnn

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semicd import SemiSegDataset
from model.semseg.segformer import SegFormer

from supervised import evaluate
from util.utils import count_params, init_log, AverageMeter
from util.dist_helper import setup_distributed

try:
    from util.apr_utils import apr_mix
except ImportError:
    print("Warning: 'util.apr_utils' not found. APR functionality will fail if triggered.")


parser = argparse.ArgumentParser()

parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--strict-deterministic', action='store_true')

# APR
parser.add_argument('--apr-prob', type=float, default=0.0)
parser.add_argument('--apr-patch-size', type=int, default=64)
parser.add_argument('--apr-ratio-factor', type=float, default=0.2)

# Boundary schedule
parser.add_argument('--bdy-start-epoch', type=int, default=10,
                    help='Start boundary loss + gate at this epoch')
parser.add_argument('--bdy-ramp-epochs', type=int, default=10,
                    help='Ramp boundary lambda & gate_scale to final over this many epochs')
parser.add_argument('--gate-scale-final', type=float, default=0.5,
                    help='Final gate_scale after ramp')


class EMATracker:
    def __init__(self, momentum=0.999):
        self.momentum = momentum
        self.val = 0.0
        self.initialized = False

    def update(self, new_val):
        if not self.initialized:
            self.val = new_val
            self.initialized = True
        else:
            self.val = self.momentum * self.val + (1 - self.momentum) * new_val
        return self.val

    def state_dict(self):
        return {'val': self.val, 'initialized': self.initialized}

    def load_state_dict(self, state):
        self.val = state.get('val', 0.0)
        self.initialized = state.get('initialized', False)


def set_seed(seed: int, strict: bool = False):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = True
    if strict:
        torch.use_deterministic_algorithms(True)


def schedule_scalar(epoch: int, start_epoch: int, ramp_epochs: int, final_value: float) -> float:
    """
    0 before start_epoch, then linear ramp to final_value over ramp_epochs.
    """
    if epoch < start_epoch:
        return 0.0
    if ramp_epochs <= 0:
        return final_value
    t = (epoch - start_epoch) / float(ramp_epochs)
    t = max(0.0, min(1.0, t))
    return final_value * t


def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    proc_seed = int(args.seed) + rank
    set_seed(proc_seed, strict=getattr(args, 'strict_deterministic', False))

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        os.makedirs(args.save_path, exist_ok=True)
        writer = SummaryWriter(args.save_path)

    ROAD_ID = cfg.get('road_id', 1)

    model = SegFormer(cfg)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    # Optimizer
    lr_multi = cfg.get('lr_multi', 10.0)
    param_groups = [
        {'params': model.backbone.parameters(), 'lr': cfg['lr']},
        {'params': [p for n, p in model.named_parameters() if 'backbone' not in n],
         'lr': cfg['lr'] * lr_multi}
    ]
    optimizer = AdamW(param_groups, lr=cfg['lr'], weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)

    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False,
        output_device=local_rank, find_unused_parameters=True
    )

    # Loss
    class_weights = torch.tensor(cfg.get('class_weights', [0.5, 2.0]), device=local_rank)
    criterion_l = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights).cuda(local_rank)
    criterion_u = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)
    criterion_bdy = nn.BCEWithLogitsLoss().cuda(local_rank)

    lambda_bdy_final = float(cfg.get('lambda_bdy', 0.4))
    conf_thresh = float(cfg.get('conf_thresh', 0.95))

    trainset_u = SemiSegDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                                cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = SemiSegDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                                cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiSegDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)

    g = torch.Generator()
    g.manual_seed(proc_seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True,
                               sampler=trainsampler_l, worker_init_fn=seed_worker, generator=g)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=2, drop_last=True,
                               sampler=trainsampler_u, worker_init_fn=seed_worker, generator=g)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, sampler=valsampler,
                           num_workers=1, worker_init_fn=seed_worker, generator=g)

    total_iters = len(trainloader_u) * cfg['epochs']
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iters)

    # States
    labeled_ratio_tracker = EMATracker(momentum=0.999)
    apr_active_next_epoch = False
    previous_best_iou = 0.0
    previous_best_acc = 0.0
    previous_best_f1 = 0.0
    previous_best_kappa = 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint.get('previous_best_iou', 0.0)
        previous_best_acc = checkpoint.get('previous_best_acc', 0.0)
        previous_best_f1 = checkpoint.get('previous_best_f1', 0.0)
        previous_best_kappa = checkpoint.get('previous_best_kappa', 0.0)
        if 'road_prior_ema' in checkpoint:
            labeled_ratio_tracker.load_state_dict(checkpoint['road_prior_ema'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])

        apr_active_next_epoch = checkpoint.get('apr_active', checkpoint.get('abd_active', False))
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        current_epoch_apr_active = apr_active_next_epoch

        cur_lambda_bdy = schedule_scalar(
            epoch=epoch,
            start_epoch=args.bdy_start_epoch,
            ramp_epochs=args.bdy_ramp_epochs,
            final_value=lambda_bdy_final
        )
        cur_gate_scale = schedule_scalar(
            epoch=epoch,
            start_epoch=args.bdy_start_epoch,
            ramp_epochs=args.bdy_ramp_epochs,
            final_value=float(args.gate_scale_final)
        )

        if rank == 0:
            logger.info(
                f'===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]["lr"]:.7f}, '
                f'BestIoU: {previous_best_iou:.2f}, APR: {current_epoch_apr_active}, '
                f'lambda_bdy: {cur_lambda_bdy:.4f}, gate_scale: {cur_gate_scale:.4f}'
            )

        total_epoch_pred_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x, boundary_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):

            img_x = img_x.cuda(local_rank, non_blocking=True)
            mask_x = mask_x.cuda(local_rank, non_blocking=True)
            boundary_x = boundary_x.cuda(local_rank, non_blocking=True)

            img_u_w = img_u_w.cuda(local_rank, non_blocking=True)
            img_u_s1 = img_u_s1.cuda(local_rank, non_blocking=True)
            img_u_s2 = img_u_s2.cuda(local_rank, non_blocking=True)

            ignore_mask = ignore_mask.cuda(local_rank, non_blocking=True)
            cutmix_box1 = cutmix_box1.cuda(local_rank, non_blocking=True)
            cutmix_box2 = cutmix_box2.cuda(local_rank, non_blocking=True)

            img_u_w_mix = img_u_w_mix.cuda(local_rank, non_blocking=True)
            img_u_s1_mix = img_u_s1_mix.cuda(local_rank, non_blocking=True)
            img_u_s2_mix = img_u_s2_mix.cuda(local_rank, non_blocking=True)
            ignore_mask_mix = ignore_mask_mix.cuda(local_rank, non_blocking=True)

            with torch.no_grad():
                model.eval()
                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            if current_epoch_apr_active:
                img_u_s1_clean = img_u_s1.clone().detach()

            img_u_s1[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand_as(img_u_s1) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand_as(img_u_s2) == 1]

            model.train()

            preds_tuple, preds_fp_tuple = model(
                torch.cat((img_x, img_u_w)),
                True,
                gate_scale=cur_gate_scale
            )
            seg_preds, bdy_preds = preds_tuple

            num_lb = img_x.shape[0]
            num_ulb = img_u_w.shape[0]

            pred_x, pred_u_w = seg_preds.split([num_lb, num_ulb])
            bdy_x, _ = bdy_preds.split([num_lb, num_ulb])

            seg_preds_fp, _ = preds_fp_tuple
            pred_u_w_fp = seg_preds_fp[num_lb:]

            pred_u_w = pred_u_w.detach()
            prob_u_w = pred_u_w.softmax(dim=1)
            conf_u_w = prob_u_w.max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            loss_apr = torch.tensor(0.0, device=local_rank)

            with torch.no_grad():
                valid_pixels_x = (mask_x != 255).sum().item()
                if valid_pixels_x > 0:
                    gt_ratio = (mask_x == ROAD_ID).sum().float().item() / valid_pixels_x
                    labeled_ratio_tracker.update(gt_ratio)

                total_valid_u = (ignore_mask != 255).sum().item()
                batch_pred_ratio = ((mask_u_w == ROAD_ID) & (conf_u_w >= conf_thresh) & (ignore_mask != 255)).sum().item() / total_valid_u if total_valid_u > 0 else 0.0
                total_epoch_pred_ratio.update(batch_pred_ratio)

            if current_epoch_apr_active and (random.random() < args.apr_prob):
                with torch.no_grad():
                    pred_u_s_clean, _ = model(img_u_s1_clean, need_fp=False, gate_scale=cur_gate_scale)
                    prob_u_s_clean = pred_u_s_clean.detach().softmax(dim=1)

                img_apr = apr_mix(
                    img_source=img_u_w,
                    img_target=img_u_s1_clean,
                    prob_source=prob_u_w,
                    prob_target=prob_u_s_clean,
                    patch_size=args.apr_patch_size
                )

                pred_apr, _ = model(img_apr, need_fp=False, gate_scale=cur_gate_scale)

                mask_valid_apr = (conf_u_w >= conf_thresh) & (ignore_mask != 255)
                loss_apr_raw = criterion_u(pred_apr, mask_u_w)
                denom = mask_valid_apr.sum()
                if denom.item() > 0:
                    loss_apr = (loss_apr_raw * mask_valid_apr).sum() / denom

            seg_strong_all, _ = model(torch.cat((img_u_s1, img_u_s2)), need_fp=False, gate_scale=cur_gate_scale)
            pred_u_s1, pred_u_s2 = seg_strong_all.chunk(2)

            mask_u_w_cutmixed1 = mask_u_w.clone()
            conf_u_w_cutmixed1 = conf_u_w.clone()
            ignore_mask_cutmixed1 = ignore_mask.clone()

            mask_u_w_cutmixed2 = mask_u_w.clone()
            conf_u_w_cutmixed2 = conf_u_w.clone()
            ignore_mask_cutmixed2 = ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]


            loss_seg_x = criterion_l(pred_x, mask_x)
            loss_bdy_x = criterion_bdy(bdy_x.squeeze(1), boundary_x.float())
            loss_x = loss_seg_x + cur_lambda_bdy * loss_bdy_x

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = (loss_u_s1 * ((conf_u_w_cutmixed1 >= conf_thresh) & (ignore_mask_cutmixed1 != 255))).sum() / \
                        ((ignore_mask_cutmixed1 != 255).sum() + 1e-8)

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = (loss_u_s2 * ((conf_u_w_cutmixed2 >= conf_thresh) & (ignore_mask_cutmixed2 != 255))).sum() / \
                        ((ignore_mask_cutmixed2 != 255).sum() + 1e-8)

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = (loss_u_w_fp * ((conf_u_w >= conf_thresh) & (ignore_mask != 255))).sum() / \
                          ((ignore_mask != 255).sum() + 1e-8)

            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + loss_apr * 0.25) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            mask_ratio = ((conf_u_w >= conf_thresh) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()

            iters = epoch * len(trainloader_u) + i
            if rank == 0:
                if iters % 100 == 0:
                    logger.info(
                        f'Iter [{iters}] '
                        f'loss: {loss.item():.4f}  '
                        f'loss_x: {loss_seg_x.item():.4f}  '
                        f'loss_s: {(loss_u_s1.item() + loss_u_s2.item()) / 2.0:.4f}  '
                        f'loss_w_fp: {loss_u_w_fp.item():.4f}  '
                        f'loss_apr: {loss_apr.item():.4f}  '
                    f'mask_ratio: {mask_ratio:.3f}'
                    )
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x_seg', loss_seg_x.item(), iters)
                writer.add_scalar('train/loss_x_bdy', loss_bdy_x.item(), iters)
                writer.add_scalar('train/lambda_bdy', cur_lambda_bdy, iters)
                writer.add_scalar('train/gate_scale', cur_gate_scale, iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/loss_apr', loss_apr.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], iters)

        # APR trigger
        epoch_avg_pred_ratio = total_epoch_pred_ratio.avg
        trigger_threshold = max(labeled_ratio_tracker.val * args.apr_ratio_factor, 0.001)
        if (epoch >= 1) and (epoch_avg_pred_ratio > trigger_threshold):
            apr_active_next_epoch = True

        if rank == 0:
            logger.info(f'[Epoch Decision] Avg Pred: {epoch_avg_pred_ratio:.5f} vs Threshold: {trigger_threshold:.5f} -> Next APR: {apr_active_next_epoch}')
            writer.add_scalar('train/epoch_avg_pred', epoch_avg_pred_ratio, epoch)
            writer.add_scalar('train/road_prior_gt', labeled_ratio_tracker.val, epoch)

        iou_class, overall_acc, f1, kappa, mAcc = evaluate(model, valloader, cfg)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> background IoU: {:.2f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> road IoU: {:.2f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}'.format(overall_acc))
            logger.info('***** Evaluation ***** >>>> F1 Score: {:.2f}'.format(f1))
            logger.info('***** Evaluation ***** >>>> Kappa Coefficient: {:.2f}\n'.format(kappa))

            writer.add_scalar('eval/background_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/road_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)
            writer.add_scalar('eval/F1_score', f1, epoch)
            writer.add_scalar('eval/Kappa', kappa, epoch)

            is_best = iou_class[1] > previous_best_iou
            previous_best_iou = max(iou_class[1], previous_best_iou)
            if is_best:
                previous_best_acc = overall_acc
                previous_best_f1 = f1
                previous_best_kappa = kappa

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch,
                'previous_best_iou': float(previous_best_iou),
                'previous_best_acc': float(previous_best_acc),
                'previous_best_f1': float(previous_best_f1),
                'previous_best_kappa': float(previous_best_kappa),
                'road_prior_ema': labeled_ratio_tracker.state_dict(),
                'apr_active': apr_active_next_epoch,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
