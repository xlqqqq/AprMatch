import argparse
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from sklearn.metrics import f1_score, cohen_kappa_score
from dataset.semicd import SemiSegDataset
from model.semseg.segformer import SegFormer
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, cfg):

    nclass = cfg['nclass']
    device = next(model.parameters()).device
    road_id = int(cfg.get('road_id', 1))

    def _as_seg(out):
        return out[0] if isinstance(out, (tuple, list)) else out

    model_was_training = model.training
    model.eval()

    conf_mat = torch.zeros((nclass, nclass), dtype=torch.long, device=device)

    with torch.inference_mode():
        for img, mask, _ in loader:
            img = img.to(device, non_blocking=True)

            seg = _as_seg(model(img))
            pred = seg.argmax(dim=1)

            mask_np = mask.numpy()
            pred_np = pred.cpu().numpy()

            valid = mask_np != 255
            if not np.any(valid):
                continue

            gt_flat = mask_np[valid].reshape(-1).astype(np.int64)
            pd_flat = pred_np[valid].reshape(-1).astype(np.int64)

            gt_t = torch.from_numpy(gt_flat).to(device)
            pd_t = torch.from_numpy(pd_flat).to(device)

            idx = gt_t * nclass + pd_t
            cm = torch.bincount(idx, minlength=nclass * nclass).reshape(nclass, nclass)
            conf_mat += cm

    if dist.is_initialized():
        dist.all_reduce(conf_mat, op=dist.ReduceOp.SUM)

    conf = conf_mat.to(torch.double)
    tp = torch.diag(conf)
    pos_gt = conf.sum(dim=1)
    pos_pd = conf.sum(dim=0)

    # IoU per class
    union = pos_gt + pos_pd - tp
    iou_per_class = tp / union.clamp(min=1.0)
    iou_class = iou_per_class.cpu().numpy() * 100.0

    # Overall accuracy
    accuracy = (tp.sum() / conf.sum().clamp(min=1.0)).item() * 100.0

    # Mean class accuracy
    acc_per_class = tp / pos_gt.clamp(min=1.0)
    mAcc = acc_per_class.mean().item() * 100.0

    # Road F1
    fp = pos_pd - tp
    fn = pos_gt - tp
    f1_per_class = (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1.0)

    if not (0 <= road_id < nclass):
        raise ValueError(f"road_id={road_id} is out of range for nclass={nclass}")
    road_f1 = f1_per_class[road_id].item() * 100.0

    # Cohen's Kappa
    po = tp.sum() / conf.sum().clamp(min=1.0)
    pe = (pos_gt * pos_pd).sum() / (conf.sum().clamp(min=1.0) ** 2)
    kappa = ((po - pe) / (1.0 - pe).clamp(min=1e-12)).item() * 100.0

    if model_was_training:
        model.train()

    return iou_class, accuracy, road_f1, kappa, mAcc



def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'segformer': SegFormer}
    assert cfg['model'] in model_zoo.keys()
    model = model_zoo[cfg['model']](cfg)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    class_weights = torch.tensor([1.0, 10.0]).cuda(local_rank)
    criterion = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights).cuda(local_rank)

    trainset = SemiSegDataset(cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path)
    valset = SemiSegDataset(cfg['dataset'], cfg['data_root'], 'val')
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best_iou, previous_best_acc = 0.0, 0.0
    epoch = -1

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best_iou = checkpoint['previous_best_iou']
        previous_best_acc = checkpoint['previous_best_acc']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best Changed IoU: {:.2f}, Overall Accuracy: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best_iou, previous_best_acc))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            # optimizer.param_groups[0]["lr"] = lr
            # optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        
        iou_class, overall_acc = evaluate(model, valloader, cfg)

        if rank == 0:
            logger.info('***** Evaluation ***** >>>> Unchanged IoU: {:.2f}'.format(iou_class[0]))
            logger.info('***** Evaluation ***** >>>> Changed IoU: {:.2f}'.format(iou_class[1]))
            logger.info('***** Evaluation ***** >>>> Overall Accuracy: {:.2f}\n'.format(overall_acc))
            
            writer.add_scalar('eval/unchanged_IoU', iou_class[0], epoch)
            writer.add_scalar('eval/changed_IoU', iou_class[1], epoch)
            writer.add_scalar('eval/overall_accuracy', overall_acc, epoch)

        is_best = iou_class[1] > previous_best_iou
        previous_best_iou = max(iou_class[1], previous_best_iou)
        if is_best:
            previous_best_acc = overall_acc
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best_iou': previous_best_iou,
                'previous_best_acc': previous_best_acc,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
