import argparse
import os

import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import time

from tqdm import tqdm

from dataloaders.kitti_loader import load_calib, input_options, KittiDepth
from metrics import AverageMeter, Result
import criteria
import helper
import vis_utils

from model import GuideFormer

# Mulit-GPU and Mixed precision supports
# NOTE : Only 1 process per GPU is supported now
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_utils import select_device
from torch.cuda import amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


os.environ["CUDA_VISIBLE_DEVICS"] = "0,1,2,3,4,5,6,7"
os.environ["MASTER_ADDR"] = 'localhost'
os.environ["MASTER_PORT"] = '12345'

LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))


def train(args, device, checkpoint=None):
    cuda = torch.cuda.is_available() and not args.cpu

    if RANK == 0: print(args)

    # Prepare train dataset
    train_dataset = KittiDepth('train', args)
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.num_gpus,
                                       rank=RANK)
    batch_size = args.batch_size // args.num_gpus

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=train_sampler,
        drop_last=False)

    # Prepare val datatset
    val_dataset = KittiDepth('val', args)
    val_sampler = DistributedSampler(val_dataset, num_replicas=args.num_gpus,
                                     rank=RANK)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=val_sampler, drop_last=False)


    # Network
    model = GuideFormer().to(device)

    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])

        if RANK == 0:
            print("=> checkpoint state loaded.")

    # Loss
    depth_criterion = criteria.MaskedMSELoss() if args.criterion == 'l2' \
        else criteria.MaskedL1Loss()

    # Optimizer and LR Scheduler
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_named_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    else:
        optimizer = torch.optim.AdamW(model_named_params, lr=args.lr, weight_decay=args.weight_decay)

    # DDP
    if cuda and RANK != -1:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[LOCAL_RANK], output_device=LOCAL_RANK)

    # logger
    logger = None
    if RANK == 0:
        logger = helper.logger(args)
        with open(os.path.join(helper.get_folder_name(args), 'hyperparams.txt'), 'w') as f:
            f.write(str(args))
        f.close()
        if checkpoint is not None:
            logger.best_result = checkpoint['best_result']
            del checkpoint
        print("=> logger created.")

    for epoch in range(args.start_epoch, args.epochs + 1):

        ### Train ###
        model.train()
        lr = helper.adjust_learning_rate(args.lr, optimizer, epoch, args)

        results_total = [torch.zeros(15, dtype=torch.float32).to(device)
                         for _ in range(args.num_gpus)]

        average_part = AverageMeter()

        train_sampler.set_epoch(epoch)

        if RANK == 0:
            print(f'===> Epoch {epoch} / {args.epochs} | lr : {lr}')

        num_sample = len(train_loader) * train_loader.batch_size * args.num_gpus

        if RANK == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        for batch, sample in enumerate(train_loader):
            # if batch >= 100: break

            dstart = time.time()

            batch_data = {key: val.to(device) for key, val in sample.items()}
            gt = batch_data['gt'].to(device)

            data_time = time.time() - dstart

            cbd_loss, dbd_loss, loss = 0, 0, 0
            w_cbd, w_dbd = 0, 0
            round1, round2 = 1, 3
            if(epoch <= round1):
                w_cbd, w_dbd = 0.2, 0.2
            elif(epoch <= round2):
                w_cbd, w_dbd = 0.05, 0.05
            else:
                w_cbd, w_dbd = 0, 0

            start = time.time()

            optimizer.zero_grad()

            cbd_pred, dbd_pred, pred = model(batch_data)
            depth_loss = depth_criterion(pred, gt)
            cbd_loss = depth_criterion(cbd_pred, gt)
            dbd_loss = depth_criterion(dbd_pred, gt)
            loss = (1 - w_cbd - w_dbd) * depth_loss + w_cbd * cbd_loss + w_dbd * dbd_loss

            loss.backward()
            optimizer.step()

            gpu_time = time.time() - start

            with torch.no_grad():
                result = Result()
                result.evaluate(pred.data, gt.data)
                average_part.update(result, gpu_time, data_time, batch_size)
            if RANK == 0:
                log_cnt += 1
                log_loss += loss.item()

                error_str = 'Epoch {} | Loss = {:.4f}'.format(epoch, log_loss / log_cnt)

                pbar.set_description(error_str)
                pbar.update(train_loader.batch_size * args.num_gpus)

        dist.all_gather(results_total, average_part.average().get_result().to(device))

        if RANK == 0:
            pbar.close()

            average_meter = AverageMeter()
            result_part = Result()
            for result_tensor in results_total:
                result_part.update(*result_tensor.cpu().numpy())
                average_meter.update(result_part, result_part.gpu_time, result_part.data_time)

            avg = logger.conditional_save_info('train', average_meter, epoch)
            is_best = logger.rank_conditional_save_best('train', avg, epoch)
            logger.conditional_summarize('train', avg, is_best)

        ### Validation ###
        torch.set_grad_enabled(False)
        model.eval()

        results_total = [torch.zeros(15, dtype=torch.float32).to(device) for _ in range(args.num_gpus)]

        average_part = AverageMeter()

        num_sample = len(val_loader) * val_loader.batch_size * args.num_gpus

        if RANK == 0:
            pbar = tqdm(total=num_sample)

        for batch, sample in enumerate(val_loader):
            # if batch >= 10 : break

            dstart = time.time()

            batch_data = {key: val.to(device) for key, val in sample.items()}
            gt = batch_data['gt']

            data_time = time.time() - dstart
            start = time.time()

            cbd_pred, dbd_pred, pred = model(batch_data)

            gpu_time = time.time() - start

            with torch.no_grad():
                result = Result()
                result.evaluate(pred.data, gt.data)
                average_part.update(result, gpu_time, data_time, batch_size)

            if RANK == 0:
                logger.conditional_save_img_comparison('val', batch, batch_data, pred,
                                                       epoch)
                pbar.update(val_loader.batch_size * args.num_gpus)

        # merge results from each gpu
        dist.all_gather(results_total, average_part.average().get_result().to(device))

        if RANK == 0:
            pbar.close()

            average_meter = AverageMeter()
            result_part = Result()
            for result_tensor in results_total:
                result_part.update(*result_tensor.cpu().numpy())
                average_meter.update(result_part, result_part.gpu_time, result_part.data_time)

            avg = logger.conditional_save_info('val', average_meter, epoch)
            is_best = logger.rank_conditional_save_best('val', avg, epoch)
            if is_best:
                logger.save_img_comparison_as_best('val', epoch)
            logger.conditional_summarize('val', avg, is_best)

            helper.save_checkpoint({  # save checkpoint
                'epoch': epoch,
                'model': model.module.state_dict(),
                'best_result': logger.best_result,
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, is_best, epoch, logger.output_directory)

        torch.set_grad_enabled(True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sparse-to-Dense')
    parser.add_argument('--num_gpus',
                        type=int,
                        default=8,
                        help='number of gpus')
    parser.add_argument('--workers',
                        default=8,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs',
                        default=100,
                        type=int,
                        metavar='N',
                        help='number of total epochs to run (default: 100)')
    parser.add_argument('--start-epoch',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--start-epoch-bias',
                        default=0,
                        type=int,
                        metavar='N',
                        help='manual epoch number bias(useful on restarts)')
    parser.add_argument('-c',
                        '--criterion',
                        metavar='LOSS',
                        default='l2',
                        choices=criteria.loss_names,
                        help='loss function: | '.join(criteria.loss_names) +
                             ' (default: l2)')
    parser.add_argument('-b',
                        '--batch-size',
                        default=1,
                        type=int,
                        help='mini-batch size (default: 1)')
    parser.add_argument('--optimizer',
                        default='adam',
                        type=str,
                        choices=['adam', 'adamw'],
                        help='optimizer')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=1e-4,
                        type=float,
                        metavar='LR',
                        help='initial learning rate (default 1e-5)')
    parser.add_argument('--weight-decay',
                        '--wd',
                        default=1e-6,
                        type=float,
                        metavar='W',
                        help='weight decay (default: 0)')
    parser.add_argument('--print-freq',
                        '-p',
                        default=10,
                        type=int,
                        metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--resume',
                        # default='./results/try10_distributed_no-amp_syncbn_lossx4_224x224_bs=6/model_best.pth.tar',
                        default='',
                        type=str,
                        metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--data-folder',
                        default='/resources/KITTI/kitti_depth',
                        type=str,
                        metavar='PATH',
                        help='data folder (default: none)')
    parser.add_argument('--data-folder-rgb',
                        default='/resources/KITTI/kitti_rgb',
                        type=str,
                        metavar='PATH',
                        help='data folder rgb (default: none)')
    parser.add_argument('--data-folder-save',
                        default='/resources/KITTI/submit_test/',
                        type=str,
                        metavar='PATH',
                        help='data folder test results(default: none)')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        default='rgbd',
                        choices=input_options,
                        help='input: | '.join(input_options))
    parser.add_argument('--val',
                        type=str,
                        default="select",
                        choices=["select", "full"],
                        help='full or select validation set')
    parser.add_argument('--jitter',
                        type=float,
                        default=0.1,
                        help='color jitter for images')
    parser.add_argument('--rank-metric',
                        type=str,
                        default='rmse',
                        choices=[m for m in dir(Result()) if not m.startswith('_')],
                        help='metrics for which best result is saved')

    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='PATH')
    parser.add_argument('--test', action="store_true", default=False,
                        help='save result kitti test dataset for submission')
    parser.add_argument('--cpu', action="store_true", default=False, help='run on cpu')

    # random cropping
    parser.add_argument('--not-random-crop', action="store_true", default=False,
                        help='prohibit random cropping')
    parser.add_argument('-he', '--random-crop-height', default=256, type=int, metavar='N',
                        help='random crop height')
    parser.add_argument('-w', '--random-crop-width', default=1216, type=int, metavar='N',
                        help='random crop height')


    # distributed learning
    parser.add_argument('--device',
                        default="0,1,2,3,4,5,6,7",
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--local_rank', type=int,
                        default=-1,
                        help='DDP parameter, do not modify')

    args = parser.parse_args()
    args.result = os.path.join('.', 'results')
    args.use_rgb = ('rgb' in args.input)
    args.use_d = 'd' in args.input
    args.use_g = 'g' in args.input
    args.val_h = 352  # 352
    args.val_w = 1216

    # DDP mode
    device = select_device(args.device, batch_size=args.batch_size)
    if LOCAL_RANK != -1:
        assert torch.cuda.device_count() > LOCAL_RANK
        assert args.batch_size % WORLD_SIZE == 0, '--batch-size must be multiple of CUDA device count'
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device('cuda', LOCAL_RANK)
        dist.init_process_group(backend='nccl', init_method='env://')  # distributed backend

    checkpoint = None
    if args.resume:  # optionally resume from a checkpoint
        args_new = args
        if os.path.isfile(args.resume):
            if RANK == 0:
                print("=> loading checkpoint '{}' ... ".format(args.resume),
                      end='')
            checkpoint = torch.load(args.resume, map_location=device)

            args.start_epoch = checkpoint['epoch'] + 1
            args.data_folder = args_new.data_folder
            args.val = args_new.val
            if RANK == 0:
                print("Completed. Resuming from epoch {}.".format(
                    checkpoint['epoch']))
        else:
            if RANK == 0:
                print("No checkpoint found at '{}'".format(args.resume))

    train(args, device, checkpoint)

    if WORLD_SIZE > 1 and RANK == 0:
        _ = [print('Destroying process group... ', end=''), dist.destroy_process_group(), print('Done.')]
