from __future__ import print_function, division

import os
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import CreateSyntheticDataset
from model.Metasurface import Metasurface
from utils.Camera import *
from RAFT.raft_stereo import RAFTStereo
from Image_formation.renderer import *
import scipy.io
from utils.evaluation import display_rgb_depth_map
from datetime import datetime

import matplotlib.pyplot as plt

from model.e2e import *


# from evaluate_stereo import *
# import core.stereo_datasets as datasets

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

def save_model_results(renderer, estimator, data_loader, mask, cam1, cam2, path):
    estimator.eval()
    with torch.no_grad():
        for k, data in enumerate(data_loader):
            ref_im_list = data['ref_im_list']
            depth_map_list = data['depth_im_list']
            occ_im_list = data['occ_im_list']
            normal_im_list = data['normal_im_list']

            gt = 1.0 / (depth_map_list[0].to(args.device).float() * 10)   
            synthetic_images, illum_img = renderer.render(ref_im_list, depth_map_list, occ_im_list, normal_im_list)
            # inv_depth_preds = estimator(synthetic_images, test_mode=True)
            
            # synthetic_images = [torch.stack([cam1.undistort(synthetic_images[l][b].cpu().numpy()) for b in range(B)]), torch.stack([cam2.undistort(synthetic_images[r][b].cpu().numpy()) for b in range(B)])]
            undd_synthetic_images = [torch.stack([cam1.undistort(left_synthetic_image.cpu().numpy()) for left_synthetic_image in synthetic_images[0]]),\
                                    torch.stack([cam2.undistort(right_synthetic_image.cpu().numpy()) for right_synthetic_image in synthetic_images[1]])]
                # gt = torch.stack([cam1.undistort(gt[b].permute(1,2,0).cpu().numpy()) for b in range(B)]).unsqueeze(1)
            gt = torch.stack([cam1.undistort(left_gt.cpu().numpy()) for left_gt in gt])
            flow_predictions = estimator(undd_synthetic_images, iters=args.valid_iters)
            break

    ref_image =(ref_im_list[0][0]/255).cpu().numpy()        
    input_image = (undd_synthetic_images[0][0]).cpu().numpy()
    gt_depth = gt[0].cpu().numpy()
    plt.imsave(f'debug/gt[d].png', gt[0].cpu().numpy(), cmap='gray')
    pred_depth = flow_predictions[-1][0][0].cpu().numpy()
    
    if pred_depth.ndim == 3:
        pred_depth = pred_depth.squeeze(0)
    # import pdb; pdb.set_trace()
    pred_depth = np.where(mask.cpu().numpy(), pred_depth, gt_depth)
    # cv2.imshow('debug', mask.cpu().numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('debug', gt_depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imshow('debug', pred_depth)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # import pdb; pdb.set_trace()


    display_rgb_depth_map(ref_image, input_image, gt_depth, pred_depth, path, args)

def grad_loss(output, gt):
    def one_grad(shift):
        ox = output[:,0, shift:] - output[:,0, :-shift]
        oy = output[:,0, :, shift:] - output[:,0, :, :-shift]
        gx = gt[:,0, shift:] - gt[:,0, :-shift]
        gy = gt[:,0, :, shift:] - gt[:,0, :, :-shift]
        loss = (ox - gx).abs().mean() + (oy - gy).abs().mean()
        return loss
    loss = (one_grad(1) + one_grad(2) + one_grad(3)) / 3.
    return loss

def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=1/0.3):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    

    # exclude invalid pixels and extremely large diplacements
    # mag = torch.sum(flow_gt**2, dim=1).sqrt()

    # exclude extremly large displacements
    # valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    valid = (valid > 0.5).unsqueeze(0).repeat(flow_gt.shape[0],1,1,1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        # i_loss = (flow_preds[i] - flow_gt).abs()
        i_loss = (flow_preds[i] - flow_gt)**2
        i_loss = i_loss + grad_loss(flow_preds[i], flow_gt)
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=1e-8)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


class Logger:

    SUM_FREQ = 100



    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        folder_dir = os.path.join(args.log, timestamp)

        if not os.path.exists(folder_dir):
            os.makedirs(folder_dir)
            self.writer = SummaryWriter(log_dir=folder_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir='runs')

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    
    device = torch.device(args.device)

    metasurface = Metasurface(args, device)
    optimized_phase = scipy.io.loadmat(args.pattern_path)['phasemap']
    metasurface.update_phase(torch.from_numpy(optimized_phase).float().to(device))

    fisheye_mask = torch.from_numpy(np.load("./fisheye_mask.npy")).to(device) 
    

    radian_90 = math.radians(90)

    cam1 = FisheyeCam(args, (0.05, 0.05, 0), (radian_90, 0, 0), 'cam1', device, args.cam_config_path)
    cam2 = FisheyeCam(args, (-0.05, 0.05, 0), (radian_90, 0, 0), 'cam2', device, args.cam_config_path)
    undd_mask = cam1.cut_and_resize(cam1.undd_valid.cpu().numpy(), cam1.undd_valid.cpu().numpy())
   
    # undd_mask = cv2.resize(cam1.undd_valid.cpu().numpy(),  (args.undd_resolution_x, args.undd_resolution_y), interpolation=cv2.INTER_LINEAR)
    
    undd_mask = torch.from_numpy(undd_mask).to(device)
    undd_mask = (undd_mask/255 > 0.5).to(torch.bool)
    # cv2.imshow('debug', undd_mask.cpu().numpy())
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # print(fisheye_mask.shape)
    # print(undd_mask.shape)
    # Front-back / in training time, we just trained cam1-cam2 front system.
    cam_calib = [cam1, cam2]

    renderer = ActiveStereoRenderer(args, metasurface, cam_calib, device)
    estimator = RAFTStereo(cam_calib, args)

    train_loss_list = []
    test_loss_list = []
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_dir = os.path.join(args.log, timestamp)

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

    dataset_train = CreateSyntheticDataset(args.train_path, 'train')
    dataset_test = CreateSyntheticDataset(args.valid_path, 'valid')
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
    dataloader_valid = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False)

    optimizer, scheduler = fetch_optimizer(args, estimator)
    scaler = GradScaler(enabled=args.mixed_precision)

    iters = 0

    logger = Logger(estimator, scheduler)

    train_loss_list = []
    test_loss_list = []

    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    folder_dir = os.path.join(args.log, timestamp)

    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)
    
    writer = SummaryWriter(log_dir=folder_dir)

    if args.restore_ckpt is not None:
        assert args.restore_ckpt.endswith(".pth")
        logging.info("Loading checkpoint...")
        checkpoint = torch.load(args.restore_ckpt)
        estimator.load_state_dict(checkpoint, strict=True)
        logging.info(f"Done loading checkpoint")

    estimator.to(device)
    estimator.train()
    estimator.freeze_bn()
    
    
    # save_model_results(renderer, estimator, dataloader_valid, os.path.join(folder_dir, "model_results_epoch0.png"))
    # save_model_results(renderer, estimator, dataloader_valid, undd_mask, cam1, cam2, os.path.join(folder_dir, "model_results_epoch0.png"))
    save_model_results(renderer, estimator, dataloader_valid, undd_mask, cam1, cam2, os.path.join(folder_dir, "model_results_epoch0.png"))

    estimator.train()
    estimator.freeze_bn()
    for epoch in range(1,1001):
        losses = []
        
        # estimator.freeze_bn() # We keep BatchNorm frozen
        # minibatch

        for i, data in enumerate(dataloader_train):
            B = args.batch_size
            
            ref_im_list = data['ref_im_list']
            depth_map_list = data['depth_im_list']
            occ_im_list = data['occ_im_list']
            normal_im_list = data['normal_im_list']


            gt = 1.0 / (depth_map_list[0].to(device).float() * 10)     
            synthetic_images, illum_img = renderer.render(ref_im_list, depth_map_list, occ_im_list, normal_im_list)
            # import pdb; pdb.set_trace()
            
            
            # synthetic_images = [torch.stack([cam1.undistort(synthetic_images[l][b].cpu().numpy()) for b in range(B)]), torch.stack([cam2.undistort(synthetic_images[r][b].cpu().numpy()) for b in range(B)])]
            undd_synthetic_images = [torch.stack([cam1.undistort(left_synthetic_image.cpu().numpy()) for left_synthetic_image in synthetic_images[0]]),\
                                    torch.stack([cam2.undistort(right_synthetic_image.cpu().numpy()) for right_synthetic_image in synthetic_images[1]])]
            gt = torch.stack([cam1.undistort(left_gt.cpu().numpy()) for left_gt in gt]).unsqueeze(1)
            
            # if epoch == 1:
            #     for it in range(gt.shape[0]):
            #         np_g = gt[it].squeeze(0).detach().cpu().numpy()
            #         plt.imsave(f'debug/gt[{k}].png', np_g, cmap='gray')
            #         k += 1
            #         import pdb; pdb.set_trace()
                    

                # import pdb; pdb.set_trace()

                # 창이 닫히지 않도록 대기
            
            optimizer.zero_grad()

            assert estimator.training
            flow_predictions = estimator(undd_synthetic_images, iters=args.train_iters)
            assert estimator.training
            loss, metrics = sequence_loss(flow_predictions, gt, undd_mask)
            # logger.writer.add_scalar("live_loss", loss.item(), global_batch_num)
            # logger.writer.add_scalar(f'learning_rate', optimizer.param_groups[0]['lr'], global_batch_num)
            # global_batch_num += 1
            print("[{0}th iter] loss : {1}".format(i, loss.item()))
            losses.append(loss.item())
            writer.add_scalar('TrainLossPerIter', loss.item(), iters)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(estimator.parameters(), 1.0)

            scaler.step(optimizer)
            scheduler.step()
            scaler.update()


            # logger.push(metrics)

            # if iters % validation_frequency == validation_frequency - 1:
            #     save_path = Path('checkpoints/%d_%s.pth' % (total_steps + 1, args.name))
            #     logging.info(f"Saving file {save_path.absolute()}")
            #     torch.save(estimator.state_dict(), save_path)

            #     results = validate_things(estimator.module, iters=args.valid_iters)

            #     logger.write_dict(results)

            #     estimator.train()
            #     model.module.freeze_bn()

            iters += 1

        train_loss = sum(losses) / len(losses)
        print("{0}/1000 epoch - Train loss : {1}".format(epoch,train_loss))
        train_loss_list.append(train_loss)
        writer.add_scalar('TrainLossPerEpoch', train_loss, epoch)

        # Test        
        estimator.eval() 
        losses = []
        
        with torch.no_grad():
            preds = []
            gts = []
            for j, data in enumerate(dataloader_valid):
                B = args.batch_size
            
                ref_im_list = data['ref_im_list']
                depth_map_list = data['depth_im_list']
                occ_im_list = data['occ_im_list']
                normal_im_list = data['normal_im_list']
                gt = 1.0 / (depth_map_list[0].to(device).float() * 10)
                synthetic_images, illum_img =  renderer.render(ref_im_list, depth_map_list, occ_im_list, normal_im_list)
                for l, r in [(0, 1)]:
                    # synthetic_images = [torch.stack([cam1.undistort(synthetic_images[l][b]) for b in range(B)]), torch.stack([cam2.undistort(synthetic_images[r][b]) for b in range(B)])]
                    undd_synthetic_images = [torch.stack([cam1.undistort(left_synthetic_image.cpu().numpy()) for left_synthetic_image in synthetic_images[l]]), torch.stack([cam2.undistort(right_synthetic_image.cpu().numpy()) for right_synthetic_image in synthetic_images[r]])]
                gt = torch.stack([cam1.undistort(left_gt.cpu().numpy()) for left_gt in gt]).unsqueeze(1)

                flow_predictions = estimator(undd_synthetic_images, iters=args.valid_iters)

                # results = validate_things(estimator.module, iters=args.valid_iters)
                
                loss, metrics = sequence_loss(flow_predictions, gt, undd_mask)
                losses.append(loss.item())

                if epoch % args.save_freq == 0:
                    if flow_predictions[-1].ndim == 4:
                        preds.extend(flow_predictions[-1].squeeze(1)[:,undd_mask].cpu().numpy().flatten())
                    else:
                        preds.extend(flow_predictions[-1][:,undd_mask].cpu().numpy().flatten())
                    if gt.ndim == 4:
                        gts.extend(gt.squeeze(1)[:,undd_mask].cpu().numpy().flatten())
                    else:
                        gts.extend(gt[:,undd_mask].cpu().numpy().flatten())
            
            test_loss = sum(losses) / len(losses)
            test_loss_list.append(test_loss)
            print("[{0}/1000 epoch - validation loss : {1}".format(epoch, sum(losses)/len(losses)))
            save_path = Path('checkpoints/%d_%s.pth' % (epoch + 1, args.name))
            # logging.info(f"Saving file {save_path.absolute()}")
            torch.save(estimator.state_dict(), save_path)
            writer.add_scalar('TestLossPerEpoch', test_loss, epoch)
       
        
        
        if epoch % args.save_freq == 0:
            save_model_results(renderer, estimator, dataloader_valid, undd_mask, cam1, cam2, os.path.join(folder_dir, "model_results_epoch%d.png"%(epoch)))
            preds = np.array(preds)
            gts = np.array(gts)
            
            mae = np.mean(np.abs(gts - preds))
            rmse = np.sqrt(np.mean((gts - preds)**2))
            abs_rel = np.mean(np.abs(gts - preds) / gts)
            sq_rel = np.mean(((gts - preds) ** 2) / gts)
            eps = 1e-5
            delta1 = np.mean((np.maximum(gts / (preds+eps), preds / (gts+eps)) < 1.25).astype(np.float64))
            delta2 = np.mean((np.maximum(gts / (preds+eps), preds / (gts+eps)) < 1.25 ** 2).astype(np.float64))
            delta3 = np.mean((np.maximum(gts / (preds+eps), preds / (gts+eps)) < 1.25 ** 3).astype(np.float64))
            
            metrics = {'mae' : mae, 'rmse' : rmse, 'abs_rel' : abs_rel, 'sq_rel' : sq_rel, 'delta1' : delta1, 'delta2' : delta2, 'delta3' : delta3}
            print(f"MAE : {metrics['mae']:.3f}, RMSE : {metrics['rmse']:.3f}, ABS_REL : {metrics['abs_rel']:.3f}, SQ_REL : {metrics['sq_rel']:.3f}, DELTA1 : {metrics['delta1']:.3f}, DELTA2 : {metrics['delta2']:.2f}, DELTA3 : {metrics['delta3']:.3f} ")
            torch.save(estimator.state_dict(), os.path.join(folder_dir, "model_epoch%d.pth"%(epoch)))
            for key, value in metrics.items():
                writer.add_scalar(f'metrics/{key}', value, epoch)

        estimator.train()
        estimator.freeze_bn()
    
    print("FINISHED TRAINING")
    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='raft-stereo', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2, help="batch size used during training.")
    parser.add_argument('--train_datasets', nargs='+', default=['Structured-light'], help="training datasets.")
    parser.add_argument('--lr', type=float, default=0.00008, help="max learning rate.")
    parser.add_argument('--num_steps', type=int, default=100000, help="length of training schedule.")
    parser.add_argument('--image_size', type=int, nargs='+', default=[400, 640], help="size of the random image crops used during training.")
    # parser.add_argument('--image_size', type=int, nargs='+', default=[160, 200], help="size of the random image crops used during training.")

    # parser.add_argument('--train_iters', type=int, default=12, help="number of updates to the disparity field in each forward pass.")
    parser.add_argument('--train_iters', type=int, default=12, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--wdecay', type=float, default=.00001, help="Weight decay in optimizer.")

    # Validation parameters
    # parser.add_argument('--valid_iters', type=int, default=12, help='number of flow-field updates during validation forward pass')
    parser.add_argument('--valid_iters', type=int, default=12, help='number of flow-field updates during validation forward pass')


    # Architecure choices
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")


    # resolution for rendering
    parser.add_argument('--fisheye_resolution_x', type=int, default=640)
    parser.add_argument('--fisheye_resolution_y', type=int, default=400)

    parser.add_argument('--pano_resolution_x', type=int, default=800)
    parser.add_argument('--pano_resolution_y', type=int, default=400)
    # parser.add_argument('--undd_resolution_x', type=int, default=664)
    # parser.add_argument('--undd_resolution_y', type=int, default=572)
    parser.add_argument('--undd_resolution_x', type=int, default=544)
    parser.add_argument('--undd_resolution_y', type=int, default=468)
    # for Camera calibaration
    parser.add_argument('--fov', type=float, default=185.0)
    parser.add_argument('--focal_length', type=float, default=1.80)
    parser.add_argument('--sensor_width', type=float, default=6.17) # 1/2.3 inch
    parser.add_argument('--sensor_height', type=float, default=4.55)
    parser.add_argument('--baseline', type=float, default=0.1) # 10cm
    
    # for Metasurface
    parser.add_argument('--N_phase', type=int, default=1000)
    parser.add_argument('--N_supercell', type=int, default=10)
    parser.add_argument('--N_theta', type=int, default=300)
    parser.add_argument('--N_alpha', type=int, default=100)
    parser.add_argument('--wave_length', type=float, default=532e-9) # mono-chromatic structured light
    parser.add_argument('--pixel_pitch', type=float, default=260e-9) # Metasurface pixel pitch

    # for image formation 
    parser.add_argument('--device', type=str, default="cuda")
    # parser.add_argument('--ambient_light_off', type=bool, default=False)
    parser.add_argument('--ambient_light_off', type=bool, default=True)

    parser.add_argument('--noise_gaussian_stddev', type=float, default=2e-2)
    # parser.add_argument('--ambient_power_max', type=float, default=0.6)
    # parser.add_argument('--ambient_power_min', type=float, default=0.6)
    parser.add_argument('--ambient_power_max', type=float, default=0.12)
    parser.add_argument('--ambient_power_min', type=float, default=0.12)
    # parser.add_argument('--laser_power_min', type=float, default=1e-1, help='previous default: 5e-1')
    # parser.add_argument('--laser_power_max', type=float, default=1.5e-0, help='previous default: 5e-1')
    parser.add_argument('--laser_power_min', type=float, default=1000, help='previous default: 5e-1')
    parser.add_argument('--laser_power_max', type=float, default=1000, help='previous default: 5e-1')

    parser.add_argument('--cam_config_path', type=str, default="./calib_results.txt")

    # Data augmentation
    parser.add_argument('--img_gamma', type=float, nargs='+', default=None, help="gamma range")
    parser.add_argument('--saturation_range', type=float, nargs='+', default=None, help='color saturation')
    parser.add_argument('--do_flip', default=False, choices=['h', 'v'], help='flip the images horizontally or vertically')
    parser.add_argument('--spatial_scale', type=float, nargs='+', default=[0, 0], help='re-scale the images randomly')
    parser.add_argument('--noyjitter', action='store_true', help='don\'t simulate imperfect rectification')
    
    # for stereo matching
    # parser.add_argument('--N_depth_candidate', type=int, default=90)
    parser.add_argument('--max_depth', type=float, default=5.0) # unit: [m]
    parser.add_argument('--min_depth', type=float, default=0.3)

    # for test 
    parser.add_argument('--num_gpu', type=int, default=1)
    parser.add_argument('--pattern_path', type=str, default="./checkpoint/pattern.mat") 
    parser.add_argument('--test_path', type=str, default='./data/test') 
    parser.add_argument('--single_img', type=bool, default=True)
    parser.add_argument('--test_save_path', type=str, default='./log/inference') 
    parser.add_argument('--chk_path', type=str, default='./checkpoint/model.pth')
    parser.add_argument('--front_right_config', type=str, default='./front_cam.npy')
    parser.add_argument('--back_right_config', type=str, default='./back_cam.npy')
    parser.add_argument('--save_freq', type=int, default=2)

    # path
    parser.add_argument('--log', type=str, default="./log/") 
    parser.add_argument('--train_path', type=str, default="./data/train")
    parser.add_argument('--valid_path', type=str, default="./data/test")

    args = parser.parse_args()




    torch.manual_seed(1234)
    np.random.seed(1234)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    Path("checkpoints").mkdir(exist_ok=True, parents=True)

    train(args)