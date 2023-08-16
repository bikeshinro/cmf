import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results, tensor2imgs
from mmdet.models.detectors.base import *


def calculate_uncertainty(cfg, model, data_loader, return_box=False):
    model.eval()
    model.cuda()
    dataset = data_loader.dataset
    print('>>> Computing Instance Uncertainty...')
    uncertainty = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())
    uncertainty_loc = torch.zeros(len(dataset)).cuda(torch.cuda.current_device())
    rsuM = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'][0] = data['img'][0].cuda()
            data.update({'x': data.pop('img')})
            y_head_f_1, y_head_f_2, y_head_f_3, y_head_f_4, y_head_f_r_1, y_head_f_r_2, y_head_f_r_3, y_head_f_r_4, y_head_cls = model(return_loss=False, rescale=True, return_box=return_box, **data)
            y_head_f_1 = torch.cat(y_head_f_1, 0)
            y_head_f_2 = torch.cat(y_head_f_2, 0)
            y_head_f_3 = torch.cat(y_head_f_3, 0)
            y_head_f_4 = torch.cat(y_head_f_4, 0)
            y_head_f_1 = torch.nn.Sigmoid()(y_head_f_1)
            y_head_f_2 = torch.nn.Sigmoid()(y_head_f_2)
            y_head_f_3 = torch.nn.Sigmoid()(y_head_f_3)
            y_head_f_4 = torch.nn.Sigmoid()(y_head_f_4)

            y_head_f_r_1 = torch.cat(y_head_f_r_1, 0)
            y_head_f_r_2 = torch.cat(y_head_f_r_2, 0)
            y_head_f_r_3 = torch.cat(y_head_f_r_3, 0)
            y_head_f_r_4 = torch.cat(y_head_f_r_4, 0)

            #we have the option to find the mean or the max of all four values. We choose to use max for this operation.
            #this is for classifier to get the epistimec uncertainty igonoring the aleatoric uncertainity for now : assuming sigma is 0
            y_head_f_avg = 0.25 * (y_head_f_1 + y_head_f_2 + y_head_f_3 + y_head_f_4)
            #y_head_f_avg = list(map(lambda m,n,o,p: (m+n+o+p)/4,y_head_f_1,y_head_f_2,y_head_f_3,y_head_f_4))
            loss_l2_p= ((y_head_f_1 - y_head_f_avg).pow(2) + (y_head_f_2 - y_head_f_avg).pow(2) + (y_head_f_3 - y_head_f_avg).pow(2) + (y_head_f_4 - y_head_f_avg).pow(2))/4

            #this is for localizer to get the epistemic uncertainty igonoring the aleatoric uncertainity for now : assuming sigma is 0
            y_head_f_r_avg = 0.25 * (y_head_f_r_1 + y_head_f_r_2 + y_head_f_r_3 + y_head_f_r_4)
            loss_l2__r_p = ((y_head_f_r_1 - y_head_f_r_avg).pow(2) + (y_head_f_r_2 - y_head_f_r_avg).pow(2) + (y_head_f_r_3 - y_head_f_r_avg).pow(2) + (y_head_f_r_4 - y_head_f_r_avg).pow(2))/4

            #loss_l2_p = (y_head_f_1 - y_head_f_2).pow(2) + (y_head_f_3 - y_head_f_4).pow(2)
            #loss_l2_p = (y_head_f_1 - y_head_f_2).pow(2)
            
            uncertainty_all_N, _ = loss_l2_p.max(dim=1) #uncertainty for classifier
            uncertainty_loc__all_N = loss_l2__r_p.max(dim=1)[0] #uncertainty for localizer, gets the maximum value over the 4 bounding box outputs
            #uncertainty_all_N = loss_l2_p.mean(dim=1)
            _, arg = uncertainty_all_N.argsort()
            uncertainty_single = uncertainty_all_N[arg[-cfg.k:]].max()
            #uncertainty_single = uncertainty_all_N[arg[-cfg.k:]].mean()
            uncertainty[i] = uncertainty_single
            
            #batch_uncertainty_loc_all_N.append(uncertainty_loc__all_N)
            uncertainty_loc[i] = uncertainty_loc__all_N.max() #max of uij over all j
            rsuM += uncertainty_loc[i]
            #rvaR += loss_l2__r_p.max()
            if i % 1000 == 0:
                print('>>> ', i, '/', len(dataset))
    overall_mean = rsuM / len(dataset)
    overall_std = (uncertainty_loc - overall_mean).square().sum/len(dataset)
    #overall_std = torch.sum(torch.square(uncertainty_loc - overall_mean))/len(dataset) #alternative method
    normalized_uncertainty_loc = (uncertainty_loc - overall_mean) / overall_std
    #arg = normalized_uncertainty_loc.argsort()
    #uncertainty_single_loc = normalized_uncertainty_loc[arg[-cfg.k:]].max()
    #uncertainty_loc[i] = uncertainty_single_loc
    aggregate_uncertainty = normalized_uncertainty_loc + uncertainty
    return aggregate_uncertainty.cpu()


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    y_heads = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data.update({'x': data.pop('img')})
            y_head = model(return_loss=False, rescale=True, **data)
        y_heads.append(y_head)
        batch_size = len(data['img_metas'][0].data)
        for _ in range(batch_size):
            prog_bar.update()
    return y_heads


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the y_heads
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes y_heads to gpu tensors and use gpu communication for y_heads
    collection. On cpu mode it saves the y_heads on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary y_heads from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect y_heads.

    Returns:
        list: The prediction y_heads.
    """
    model.eval()
    y_heads = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data.update({'x': data.pop('img')})
            y_head = model(return_loss=False, rescale=True, **data)
            # encode mask y_heads
        y_heads.append(y_head)
        if rank == 0:
            batch_size = (len(data['img_meta'].data) if 'img_meta' in data else len(data['img_metas'][0].data))
            for _ in range(batch_size * world_size):
                prog_bar.update()
    # collect y_heads from all ranks
    if gpu_collect:
        y_heads = collect_y_heads_gpu(y_heads, len(dataset))
    else:
        y_heads = collect_y_heads_cpu(y_heads, len(dataset), tmpdir)
    return y_heads


def collect_y_heads_cpu(y_head_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part y_head to the dir
    mmcv.dump(y_head_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load y_heads of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the y_heads
        ordered_y_heads = []
        for res in zip(*part_list):
            ordered_y_heads.extend(list(res))
        # the dataloader may pad some samples
        ordered_y_heads = ordered_y_heads[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_y_heads


def collect_y_heads_gpu(y_head_part, size):
    rank, world_size = get_dist_info()
    # dump y_head part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(y_head_part)), dtype=torch.uint8, device='cuda')
    # gather all y_head part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding y_head part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all y_head part
    dist.all_gather(part_recv_list, part_send)
    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the y_heads
        ordered_y_heads = []
        for res in zip(*part_list):
            ordered_y_heads.extend(list(res))
        # the dataloader may pad some samples
        ordered_y_heads = ordered_y_heads[:size]
        return ordered_y_heads
