import argparse
import yaml
import shutil
import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Union, Tuple, List
import os
from glob import glob
from scipy.ndimage.filters import gaussian_filter
from typing import Tuple
from tqdm import tqdm
from nn_transunet.default_configuration import get_default_configuration
from nn_transunet.configuration import default_plans_identifier
from nn_transunet.networks.nnunet_model import Generic_UNet
from nn_transunet.utils.dist_utils import download_from_hdfs
from batchgenerators.utilities.file_and_folder_operations import *
from collections import OrderedDict
from nn_transunet.networks.neural_network import no_op
from torch.cuda.amp import autocast
import cc3d
import pandas as pd
from nn_transunet.networks.transunet3d_model import InitWeights_He
import threading
from filelock import FileLock

thread_lock = threading.Lock()  # NOTE: all threads shared the same threading lock
class FileLockForAllUsers(FileLock):
    def _acquire(self):
        # 调用父类方法创建并加锁
        super()._acquire()
        # 将锁文件权限修改为 777，确保所有用户都能访问
        try:
            os.chmod(self.lock_file, 0o777)
        except Exception as e:
            print(f"\033[31mError found when giving lock file 777 permission: {e}\033[0m")
            # 这里可以记录日志，处理失败情况
            pass

def get_flops(model, test_data):
    from flop_count.flop_count import flop_count
    batch_size = test_data.shape[0]
    flop_dict, _ = flop_count(model, (test_data,))
    msg = 'model_flops' + '\t' + str(sum(flop_dict.values()) / batch_size) + 'G'+ '\t params:' + str(
        sum([m.numel() for m in model.parameters()])) + '\n-----------------'
    return msg


parser = argparse.ArgumentParser()
parser.add_argument('--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')
parser.add_argument("--fold", default=0, help='0, 1, ..., 5 or \'all\'')
parser.add_argument("--raw_data_dir", default='')
parser.add_argument("--raw_data_folder", default='imagesTr', help='can be imagesVal')
parser.add_argument("--save_folder", default=None)
parser.add_argument("--save_npz", default=False, action="store_true")
parser.add_argument("--disable_split", default=False, action="store_true", help='just use raw_data_dir, do not use split!')
parser.add_argument("--model_latest", default=False, action="store_true", help='')
parser.add_argument("--model_final", default=False, action="store_true", help='')
parser.add_argument("--model_file", default=None, type=str)
parser.add_argument("--mixed_precision", default=True, type=bool, help='')
parser.add_argument("--measure_param_flops", default=False, action="store_true", help='')

##################### the args from train script #########################################
parser.add_argument("--local_rank", type=int, default=0)
parser.add_argument('--loss_name', default='', type=str)
parser.add_argument('--plan_update', default='', type=str)
parser.add_argument('--crop_size', nargs='+', type=int, default=None,
                    help='input to network')
parser.add_argument("--pretrained", default=False, action="store_true", help="")
parser.add_argument("--disable_decoder", default=False, action="store_true", help="disable decoder of mae network")
parser.add_argument("--model_params", default={})
parser.add_argument('--layer_decay', default=1.0, type=float, help="layer-wise dacay for lr")
parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                    help='Drop path rate (default: 0.1), drop_path=0 for MAE pretrain')
parser.add_argument("--find_zero_weight_decay", default=False, action="store_true", help="")
parser.add_argument('--n_class', default=17, type=int, help="")
parser.add_argument('--deep_supervision_scales', nargs='+', type=int, default=[], help='')
parser.add_argument("--fix_ds_net_numpool", default=False, action="store_true", help="")
parser.add_argument("--num_ds", default=None, type=int, help="")
parser.add_argument("--is_sigmoid", default=False, action="store_true", help="")
parser.add_argument("--num_examples", type=int, help="")

parser.add_argument('--input_csv', type=str, required=True,
                    help='Input CSV file. Should contains `Original ID` and `BDMAP ID` columns')    
parser.add_argument('--output_csv', type=str, required=True,
                    help='Output CSV file. Normally the same csv file name as input') 
parser.add_argument('--output_folder_name', type=str, required=True) 

args, remaining = parser.parse_known_args()
model_params = {}
if args.config:
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
        parser.set_defaults(**cfg)
    if "model_params" in cfg.keys():
        model_params = cfg["model_params"]
else:
    raise NotImplementedError

args = parser.parse_args()


network, task, network_trainer, hdfs_base = cfg['network'], cfg['task'], cfg['network_trainer'], cfg['hdfs_base']

plans_identifier = default_plans_identifier
plans_file, output_folder_name, dataset_directory, batch_dice, stage, \
        trainer_class = get_default_configuration(network, task, network_trainer, plans_identifier, hdfs_base=hdfs_base, plan_update='' if 'plan_update' not in cfg.keys() else cfg['plan_update'])
os.makedirs(output_folder_name, exist_ok=True)

fold_name = args.fold if isinstance(args.fold, str) and args.fold.startswith('all') else 'fold_'+str(args.fold)
output_folder_name = args.output_folder_name
output_folder =  output_folder_name + '/' + fold_name

plans_path = os.path.join(output_folder_name, 'plans.pkl')
shutil.copy(plans_file, plans_path)
print('plans_file', plans_file)
val_keys = None
if not args.disable_split:
    splits_file = os.path.join(dataset_directory, "splits_final.pkl")
    splits = load_pickle(splits_file)
    if not args.fold.startswith('all'):
        assert int(args.fold) < len(splits)
        val_keys = splits[int(args.fold)]['val']
        if isinstance(val_keys, np.ndarray):
            val_keys = val_keys.tolist()


print("output folder for snapshot loading exists: ", output_folder)
prefix = "version5"
planfile = plans_path
if not args.model_file:
    if os.path.exists(output_folder + '/' + 'model_best.model') and not args.model_latest and not args.model_final:
        print("load model_best.model")
        modelfile = output_folder + '/' + 'model_best.model'
    elif os.path.exists(output_folder + '/' + 'model_final_checkpoint.model') and not args.model_latest:
        print("load model_final_checkpoint.model")
        modelfile = output_folder + '/' + 'model_final_checkpoint.model'
    else:
        print("load model_latest.model")
        modelfile = output_folder + '/' + 'model_latest.model'
else:
    print("load model from", args.model_file)
    modelfile = output_folder + '/' + args.model_file

info = pickle.load(open(planfile, "rb"))
plan_data = {}
plan_data["plans"] = info

resolution_index = 1
if cfg['task'].find('500') != -1: # multiphase task e.g, Brats
    resolution_index = 0

num_classes = plan_data['plans']['num_classes']
if args.config.find('500Region') != -1:
    regions = {"whole tumor": (1, 2, 3), "tumor core": (2, 3), "enhancing tumor": (3,) }
    regions_class_order = (1, 2, 3)
    num_classes = len(regions)
else:
    num_classes += 1 # add background

base_num_features = plan_data['plans']['base_num_features']
# 解决size mismatch，因为路径里有001出现，导致resolution_index变成0
if 'Task005' in plans_file or  'Task004' in plans_file or 'Task001' in plans_file or 'Task002' in plans_file or 'Task1006' in plans_file: # multiphase task e.g, Brats
    resolution_index = 0

patch_size = plan_data['plans']['plans_per_stage'][resolution_index]['patch_size']
patch_size = args.crop_size if args.crop_size is not None else patch_size

num_input_channels = plan_data['plans']['num_modalities']
conv_per_stage = plan_data['plans']['conv_per_stage'] if "conv_per_stage" in plan_data['plans'].keys() else 2
use_mask_for_norm = plan_data['plans']['use_mask_for_norm']
normalization_schemes = plan_data['plans']['normalization_schemes']
intensity_properties = plan_data['plans']['dataset_properties']['intensityproperties']
transpose_forward, transpose_backward = plan_data['plans']['transpose_forward'], plan_data['plans']['transpose_backward']

pool_op_kernel_sizes = plan_data['plans']['plans_per_stage'][resolution_index]['pool_op_kernel_sizes']
conv_kernel_sizes = plan_data['plans']['plans_per_stage'][resolution_index]['conv_kernel_sizes']
current_spacing = plan_data['plans']['plans_per_stage'][resolution_index]['current_spacing']

try:
    mean = plan_data['plans']['dataset_properties']['intensityproperties'][0]['mean']
    std = plan_data['plans']['dataset_properties']['intensityproperties'][0]['sd']
    clip_min = plan_data['plans']['dataset_properties']['intensityproperties'][0]['percentile_00_5']
    clip_max = plan_data['plans']['dataset_properties']['intensityproperties'][0]['percentile_99_5']
except:
    if cfg['task'].find('500') != -1 or '005' in task or '001' in task:
        mean, std, clip_min, clip_max = 0, 1, -9999, 9999
    else:
        mean, std, clip_min, clip_max = None, None, -9999, 9999

if cfg['model'].startswith('Generic'):
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    dropout_op_kwargs = {'p': 0, 'inplace': True}
    net_nonlin = nn.LeakyReLU
    net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}

    if cfg['model'] == 'Generic_TransUNet_max_ppbp':
        from nn_transunet.networks.transunet3d_model import Generic_TransUNet_max_ppbp
        net = Generic_TransUNet_max_ppbp(num_input_channels, base_num_features, num_classes, len(pool_op_kernel_sizes), conv_per_stage, 2,
                   nn.Conv3d, nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d,
                   dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, not args.disable_ds, False, lambda x: x,
                   InitWeights_He(1e-2), pool_op_kernel_sizes, conv_kernel_sizes, False, True, 
                    convolutional_upsampling= True,    
                    patch_size=patch_size, **model_params)
    elif cfg['model'] == 'Generic_UNet':
        net = Generic_UNet(num_input_channels, base_num_features, num_classes, len(pool_op_kernel_sizes), conv_per_stage, 2,
                   nn.Conv3d, nn.InstanceNorm3d, norm_op_kwargs, nn.Dropout3d,
                   dropout_op_kwargs, net_nonlin, net_nonlin_kwargs, False, False, lambda x: x,
                   InitWeights_He(1e-2), pool_op_kernel_sizes, conv_kernel_sizes, False, True, True)
    else:
        raise NotImplementedError

else:
    net = None
    print("Note implemented for cfg['model']")
    raise NotImplementedError
total = sum([param.nelement() for param in net.parameters()])
net.cuda()



if args.measure_param_flops:
    with torch.no_grad():
        test_data = torch.zeros((1, 1, patch_size[0], patch_size[1], patch_size[2])).cuda()
        msg = get_flops(net, test_data) # same flops results with detectron, but count params as well.
        print(args.config, msg)
    sys.exit(0)


checkpoint = torch.load(modelfile)

print("load epoch", checkpoint['epoch'])
new_state_dict = OrderedDict()
curr_state_dict_keys = list(net.state_dict().keys())
for k, value in checkpoint['state_dict'].items():
    key = k
    if key not in curr_state_dict_keys and key.startswith('module.'):
        key = key[7:]
    new_state_dict[key] = value

net.load_state_dict(new_state_dict, strict=False) 
cur_dict = net.state_dict()
print("missing keys of pretrained", [k for k in new_state_dict.keys() if k not in cur_dict.keys()])
print("extra keys of pretrained", [k for k in cur_dict.keys() if k not in new_state_dict.keys()])
print("weights loaded to network")
net.eval()



def _get_arr(path):
    sitkimg = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(sitkimg)
    return arr, sitkimg

def _write_arr(arr, path, info=None):
    sitkimg = sitk.GetImageFromArray(arr)
    if info is not None:
        sitkimg.CopyInformation(info)
    sitk.WriteImage(sitkimg, path)


def get_do_separate_z(spacing, anisotropy_threshold=2):
    do_separate_z = spacing[-1] > anisotropy_threshold
    return do_separate_z


def _compute_steps_for_sliding_window(patch_size: Tuple[int, ...],
                                      image_size: Tuple[int, ...],
                                      step_size: float) -> List[List[int]]:
    assert [i >= j for i, j in zip(image_size, patch_size)], "image size must be as large or larger than patch_size"
    assert 0 < step_size <= 1, 'step_size must be larger than 0 and smaller or equal to 1'
    target_step_sizes_in_voxels = [i * step_size for i in patch_size]

    num_steps = [int(np.ceil((i - k) / j)) + 1 for i, j, k in zip(image_size, target_step_sizes_in_voxels, patch_size)]

    steps = []
    for dim in range(len(patch_size)):
        max_step_value = image_size[dim] - patch_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # does not matter because there is only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]

        steps.append(steps_here)

    return steps


def _get_gaussian(patch_size, sigma_scale=1. / 8) -> np.ndarray:
    tmp = np.zeros(patch_size)
    center_coords = [i // 2 for i in patch_size]
    sigmas = [i * sigma_scale for i in patch_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(np.float32)
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map


gaussian_mask = torch.from_numpy(_get_gaussian(patch_size)[np.newaxis, np.newaxis]).cuda().half().clamp_min_(1e-4)


def predict(arr):
    if args.num_ds is not None:
        prob_map = torch.zeros((args.num_ds, 1, num_classes,) + arr.shape[-3:]).half().cuda()
    else:
        prob_map = torch.zeros((1, num_classes,) + arr.shape[-3:]).half().cuda()

    cnt_map = torch.zeros_like(prob_map)

    arr_clip = np.clip(arr, clip_min, clip_max)


    if mean is None and std is None:
        raw_norm = (arr_clip - arr_clip.mean()) / (arr_clip.std()+ 1e-8)
    else:
        raw_norm = (arr_clip - mean) / std
    
    step_size = 0.5 if args.config.find('500Region') != -1 or  task.find('005') != -1  or task.find('001')  != -1 else 0.7
    steps = _compute_steps_for_sliding_window(patch_size, raw_norm.shape[-3:], step_size) # step_size=0.5 for Brats
    
    num_tiles = len(steps[0]) * len(steps[1]) * len(steps[2])

    print("data shape:", raw_norm.shape)
    print("patch size:", patch_size)
    print("steps (x, y, and z):", steps)
    print("number of tiles:", num_tiles)

    for x in steps[0]:
        lb_x = x
        ub_x = x + patch_size[0]
        for y in steps[1]:
            lb_y = y
            ub_y = y + patch_size[1]
            for z in steps[2]:
                lb_z = z
                ub_z = z + patch_size[2]
                with torch.no_grad():
                    numpy_arr = raw_norm[:, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis] if len(raw_norm.shape)==4 else raw_norm[lb_x:ub_x, lb_y:ub_y, lb_z:ub_z][np.newaxis, np.newaxis]
                    tensor_arr = torch.from_numpy(numpy_arr).cuda().half()
                    seg_pro = net(tensor_arr) # (1, c, d, h, w)
                    if args.num_ds is not None and isinstance(seg_pro, dict) and ('is_max_hungarian' in model_params.keys() and model_params['is_max_hungarian']):
                        aux_cls_out  = [p['pred_logits'] for p in seg_pro['aux_outputs']]
                        aux_mask_out = [p['pred_masks'] for p in seg_pro['aux_outputs']]
                        all_cls_out, all_mask_out = [seg_pro["pred_logits"]] + aux_cls_out[::-1], [seg_pro["pred_masks"]] + aux_mask_out[::-1]
                        for i, (mask_cls, mask_pred) in enumerate(zip(all_cls_out, all_mask_out)): # desceding order
                            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # filter out non-object class
                            mask_pred = mask_pred.sigmoid()
                            _seg_pro = torch.einsum("bqc,bqdhw->bcdhw", mask_cls, mask_pred)
                            _pred = _seg_pro * gaussian_mask
                            prob_map[i, :, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred 

                    elif isinstance(seg_pro, dict) and ('is_max_hungarian' in model_params.keys() and model_params['is_max_hungarian']):
                        mask_cls, mask_pred = seg_pro["pred_logits"], seg_pro["pred_masks"]
                        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] # filter out non-object class
                        mask_pred = mask_pred.sigmoid()
                        seg_pro = torch.einsum("bqc,bqdhw->bcdhw", mask_cls, mask_pred)
                        _pred = seg_pro

                        if args.config.find('500Region') != -1 or  task.find('005') != -1  or task.find('001')  != -1:
                            _pred = seg_pro
                        else:
                            _pred = seg_pro * gaussian_mask

                        prob_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred
                        # NOTE: should also smooth cnt_map if apply gaussian_mask before |  neural_network.py -> network.predict_3D -> _internal_predict_3D_3Dconv_tiled
                        cnt_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += 1

                    elif args.num_ds is not None and not isinstance(seg_pro, dict): # (isinstance(seg_pro, list) or isinstance(seg_pro, tuple))
                        assert len(seg_pro) == args.num_ds, (len(seg_pro), args.num_ds)
                        for i, _seg_pro in enumerate(seg_pro):
                            if torch.sum(_seg_pro[0,:,0,0,0]) != 1: 
                                _seg_pro = torch.softmax(_seg_pro, dim=1)
                            _pred = _seg_pro * gaussian_mask
                            prob_map[i, :, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred  

                    elif isinstance(seg_pro, list) or isinstance(seg_pro, tuple):
                        seg_pro = seg_pro[0]
                        if torch.sum(seg_pro[0,:,0,0,0]) != 1: 
                            seg_pro = torch.softmax(seg_pro, dim=1)
                        _pred = seg_pro * gaussian_mask
                        prob_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred
                    else:

                        if args.is_sigmoid:
                            _pred = seg_pro.sigmoid()
                        elif torch.sum(seg_pro[0,:,0,0,0]) != 1: 
                            seg_pro = torch.softmax(seg_pro, dim=1)
                            _pred = seg_pro * gaussian_mask

                        prob_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += _pred
                        cnt_map[:, :, lb_x:ub_x, lb_y:ub_y, lb_z:ub_z] += 1 


    print("before devision", prob_map.max(), prob_map.min(), cnt_map.min())
    if args.config.find('500Region') != -1 or  task.find('005') != -1  or task.find('001')  != -1:
        prob_map /= cnt_map
    print("after devision", prob_map.max(), prob_map.min())
    # viz_data(torch.from_numpy(raw_norm[np.newaxis, np.newaxis]).cuda().half(), prob_map)
    torch.cuda.empty_cache()
    return prob_map.detach().cpu()


def itk_change_spacing(src_itk, output_spacing, interpolate_method='Linear'):
    assert interpolate_method in ['Linear', 'NearestNeighbor']
    src_size = src_itk.GetSize()
    src_spacing = src_itk.GetSpacing()

    re_sample_scale = tuple(np.array(src_spacing) / np.array(output_spacing).astype(float))
    re_sample_size = tuple(np.array(src_size).astype(float) * np.array(re_sample_scale))

    re_sample_size = [int(round(x)) for x in re_sample_size]
    output_spacing = tuple((np.array(src_size) / np.array(re_sample_size)) * np.array(src_spacing))

    re_sampler = sitk.ResampleImageFilter()
    re_sampler.SetOutputPixelType(src_itk.GetPixelID())
    re_sampler.SetReferenceImage(src_itk)
    re_sampler.SetSize(re_sample_size)
    re_sampler.SetOutputSpacing(output_spacing)
    re_sampler.SetInterpolator(eval('sitk.sitk' + interpolate_method))
    return re_sampler.Execute(src_itk)

def resample_image_to_ref(image, ref, interp=sitk.sitkNearestNeighbor, pad_value=0):
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(ref)
    resample.SetDefaultPixelValue(pad_value)
    resample.SetInterpolator(interp)
    return resample.Execute(image)

def keep_largest_component_multi_cls(seg_data, labels):
    largest_cc = np.zeros_like(seg_data)
    seg_data_mask = np.isin(seg_data, labels).astype(np.uint8)
    # print((np.isin(seg_data, labels)).shape)
    labeled, N = cc3d.connected_components(seg_data_mask==1, connectivity=26, return_N=True)
    max_component = None
    max_size = 0
    for i in range(1, N + 1):
        component = (labeled == i)
        size = np.sum(component)
        if size > max_size:
            max_component = component
            max_size = size
    if max_component is not None:
        # seg_data[seg_data_mask==1] = 0
        largest_cc[max_component] = 1
    return largest_cc

def get_shared_properties(pred_pancreas_tumor, probabilities_final, spacing, label_id):
    if label_id == 0:
        has_label = int((pred_pancreas_tumor>0).sum() > 0)
    else:
        has_label = int((pred_pancreas_tumor==label_id).sum() > 0)
    # largest_cc = np.zeros_like(seg_data)
    # seg_data_mask = np.isin(seg_data, labels).astype(np.uint8)
    # # print((np.isin(seg_data, labels)).shape)
    # breakpoint()
    labeled, N = cc3d.connected_components(pred_pancreas_tumor==label_id, connectivity=26, return_N=True)
    voxel_size_list = []
    volume_size_list = []
    largest_logits_list = []
    for i in range(1, N + 1):
        component = (labeled == i)
        voxel_size = int(np.sum(component))
        volume_size = round(float(voxel_size * spacing[0] * spacing[1] * spacing[2]), 2)
        if probabilities_final is not None:
            largest_logits = round(float(probabilities_final[label_id][component].max()), 2)
        else: 
            largest_logits = -1.    # `probabilities_final` is None
        voxel_size_list.append(voxel_size)
        volume_size_list.append(volume_size)
        largest_logits_list.append(largest_logits)
    # print(voxel_size_list, volume_size_list, largest_logits_list)

    if N > 0:
        sorted_info = sorted(zip(voxel_size_list, volume_size_list, largest_logits_list), reverse=True) # sort based on the first row
        voxel_size_list, volume_size_list, largest_logits_list = list(zip(*sorted_info))    # sort based on the fist list
        return has_label, N, list(voxel_size_list), list(volume_size_list), largest_logits_list[0]  # get only the largest logit in the largest cc
    else:
        return has_label, N, list(voxel_size_list), list(volume_size_list), 0.
    
def parse_prediction_new_row(pred_pancreas_tumor, probabilities_final, bdmap_id, shape, spacing):

    # shape = (int(properties_dict["shape_before_cropping"][1]), int(properties_dict["shape_before_cropping"][2]), int(properties_dict["shape_before_cropping"][0]))  # (C H W) -> (H W C)
    # spacing = (float(properties_dict["spacing"][1]), float(properties_dict["spacing"][2]), float(properties_dict["spacing"][0]))    # (C H W) -> (H W C)
    # shape = 
    # spacing =
    # breakpoint()
    spcaing_for_volume_size = (spacing[2], spacing[0], spacing[1]) 
    # pancreas, pancreas_component_count, pancreas_voxel_size, pancreas_volume_size, _ = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreas"])
    # duct, duct_component_count, duct_voxel_size, duct_volume_size, _ = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_duct"])
    # PDAC, PDAC_component_count, PDAC_voxel_size, PDAC_volume_size, PDAC_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_pdac"])
    # cyst, cyst_component_count, cyst_voxel_size, cyst_volume_size, cyst_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_cyst"])
    # PNET, PNET_component_count, PNET_voxel_size, PNET_volume_size, PNET_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, dataset_json_dict_or_file["labels"]["pancreatic_pnet"])

    pancreas, pancreas_component_count, pancreas_voxel_size, pancreas_volume_size, _ = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, 13)
    duct, duct_component_count, duct_voxel_size, duct_volume_size, _ = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, 14)
    PDAC, PDAC_component_count, PDAC_voxel_size, PDAC_volume_size, PDAC_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, 23)
    cyst, cyst_component_count, cyst_voxel_size, cyst_volume_size, cyst_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, 24)
    PNET, PNET_component_count, PNET_voxel_size, PNET_volume_size, PNET_largest_logit = get_shared_properties(pred_pancreas_tumor, probabilities_final, spcaing_for_volume_size, 25)

    return bdmap_id, shape, spacing, \
        pancreas,pancreas_component_count,pancreas_voxel_size,pancreas_volume_size, \
        duct,duct_component_count,duct_voxel_size,duct_volume_size, \
        PDAC,PDAC_component_count,PDAC_voxel_size,PDAC_volume_size,PDAC_largest_logit, \
        cyst,cyst_component_count,cyst_voxel_size,cyst_volume_size,cyst_largest_logit, \
        PNET,PNET_component_count,PNET_voxel_size,PNET_volume_size,PNET_largest_logit

def Inference3D_multiphase(rawf, save_path=None, mode='nii'):
    """if use nii, can crop and preprocess given a list of path; already check the npy is the same with f(nii)"""
    if mode=='npy': # preferred
        rawf_npy = rawf.replace('nnUNet_raw_data', 'nnUNet_preprocessed').replace('imagesTr', 'nnUNetData_plans_v2.1_stage0').replace('_0000.nii.gz', '.npy')
        assert os.path.exists(rawf_npy) # already preprocessed!
        img_arr = np.load(rawf_npy)[:4]
    else:
        # data_files: a list of path
        # nnunet.inference.predict will call 
        # nnunet.training.network_training.nnUNetTrainer -> nnUNetTrainer.preprocess_patient(data_files) # for new unseen data.
        # nnunet.preprocessing.preprocessing -> GenericPreprocessor.preprocess_test_case(data_files, current_spacing) will do ImageCropper.crop_from_list_of_files(data_files) and resample_and_normalize
        # return data, seg, properties
        data_files = [] # an element in lists_of_list: [[case0_0000.nii.gz, case0_0001.nii.gz], [case1_0000.nii.gz, case1_0001.nii.gz], ...]
        for i in range(num_input_channels):
            data_files.append(rawf.replace('0000', '000'+str(i)))

        from nnunet.preprocessing.cropping import ImageCropper
        from nnunet.preprocessing.preprocessing import GenericPreprocessor
        data, seg, crop_properties = ImageCropper.crop_from_list_of_files(data_files, seg_file=None) # can retrive properties['crop_bbox'], seg is all -1 array
        force_separate_z, target_spacing = False, [1.0, 1.0, 1.0]
        data, seg = data.transpose((0, *[i + 1 for i in transpose_forward])), seg.transpose((0, *[i + 1 for i in transpose_forward]))
        preprocessor = GenericPreprocessor(normalization_schemes, use_mask_for_norm, transpose_forward, intensity_properties)
        data, _, crop_prep_properties = preprocessor.resample_and_normalize(data, target_spacing, crop_properties, seg, force_separate_z=force_separate_z)
        img_arr = data
    pad_flag = 0
    padzyx = np.clip(np.array(patch_size) - np.array(img_arr.shape)[-3:], 0, 1000) # clip the shape..
    if np.any(padzyx > 0):
        pad_flag = 1
        pad_left = padzyx // 2
        pad_right = padzyx - padzyx // 2
        img_arr = np.pad(img_arr, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]), (pad_left[2], pad_right[2])))
    
    # PREDICT!
    if args.mixed_precision:
        context = autocast
    else:
        context = no_op

    with context():
        with torch.no_grad():
            prob_map = predict(img_arr)

    if pad_flag:
        prob_map = prob_map[:, :,
                   pad_left[0]: img_arr[0].shape[0] - pad_right[0],
                   pad_left[1]: img_arr[0].shape[1] - pad_right[1],
                   pad_left[2]: img_arr[0].shape[2] - pad_right[2]]
    del img_arr

    prob_map_interp = prob_map.cpu().numpy()
    if save_path is None:
        save_dir = rawf.replace(".nii.gz", "_pred.nii.gz")
    else:
        uid = rawf.split("/")[-1].replace('_0000', '')
        for i in range(num_input_channels):
            uid = uid.replace('_000'+str(i), '')
        save_dir = os.path.join(save_path, uid)


    # convert from region to normal label
    if args.config.find('500Region') != -1 or  task.find('005') != -1  or task.find('001')  != -1:
        thres = float(args.model_params['max_infer'].split('_')[-1]) if 'max_infer' in args.model_params.keys() and args.model_params['max_infer'].startswith('thres') else 0.5
        segmentation_hot = (prob_map > thres).float().squeeze(0).cpu().numpy() # (3, D, H, W)

        segmentation = np.zeros(segmentation_hot.shape[1:])
        segmentation[segmentation_hot[0]==1] = 1
        segmentation[segmentation_hot[1]==1] = 2
        segmentation[segmentation_hot[2]==1] = 3
    else:
        segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)


    if args.save_npz:
        save_npz_dir = save_path.replace("nnUNet_inference", "nnUNet_inference_npz")
        os.makedirs(save_npz_dir, exist_ok=True)
        npz_name = rawf.split("/")[-1].replace(".nii.gz", ".npz")
        np.savez_compressed(os.path.join(save_npz_dir, npz_name), softmax=prob_map_interp.squeeze(0))

    if mode=='nii':
        bbox, original_size_of_raw_data = crop_prep_properties['crop_bbox'], crop_prep_properties['original_size_of_raw_data']
        print(bbox, original_size_of_raw_data)
        full_segm = np.zeros(original_size_of_raw_data)
        if '005' in task:
            full_segm[:, 0:segmentation.shape[1], 0:segmentation.shape[2]] = segmentation[:full_segm.shape[0]]
        else:
             full_segm[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1], bbox[2][0]:bbox[2][1]] = segmentation
        segmentation = full_segm

    del prob_map_interp
    pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    sitk.WriteImage(pred_sitk, save_dir)


def Inference3D(rawf, save_path=None, output_csv_path=None):
    arr_raw, sitk_raw = _get_arr(rawf)
    origin_spacing = sitk_raw.GetSpacing()
    rai_size = sitk_raw.GetSize()
    print("origin_spacing: ", origin_spacing)
    print("raw_size", rai_size)
    if (get_do_separate_z(origin_spacing) or get_do_separate_z(current_spacing[::-1])):
        print('preprocessing: do seperate z.....')
        img_arr = []
        for i in range(rai_size[-1]):
            img_arr.append(sitk.GetArrayFromImage(itk_change_spacing(sitk_raw[:, :, i], current_spacing[::-1][:-1])))
        img_arr = np.array(img_arr)
        img_sitk = sitk.GetImageFromArray(img_arr)
        img_sitk.SetOrigin(sitk_raw.GetOrigin())
        img_sitk.SetDirection(sitk_raw.GetDirection())
        img_sitk.SetSpacing(tuple(current_spacing[::-1][:-1]) + (origin_spacing[-1],))
        img_arr = sitk.GetArrayFromImage(
            itk_change_spacing(img_sitk, current_spacing[::-1], interpolate_method='NearestNeighbor'))
    else:
        img_arr = sitk.GetArrayFromImage(itk_change_spacing(sitk_raw, current_spacing[::-1]))
    pad_flag = 0
    padzyx = np.clip(np.array(patch_size) - np.array(img_arr.shape), 0, 1000) # clip the shape..
    if np.any(padzyx > 0):
        pad_flag = 1
        pad_left = padzyx // 2
        pad_right = padzyx - padzyx // 2
        img_arr = np.pad(img_arr,
                         ((pad_left[0], pad_right[0]), (pad_left[1], pad_right[1]), (pad_left[2], pad_right[2])))

    # PREDICT!
    if args.mixed_precision:
        context = autocast
    else:
        context = no_op

    with context():
        with torch.no_grad():
            prob_map = predict(img_arr)

    if args.num_ds is not None:
        img_arr_shape = img_arr.shape
        del img_arr
        for idx in range(args.num_ds):
            _prob_map = prob_map[idx]
            if pad_flag:
                _prob_map = _prob_map[:, :,
                        pad_left[0]: img_arr_shape[0] - pad_right[0],
                        pad_left[1]: img_arr_shape[1] - pad_right[1],
                        pad_left[2]: img_arr_shape[2] - pad_right[2]]
            

            if (get_do_separate_z(origin_spacing) or get_do_separate_z(current_spacing[::-1])):
                print('postpreprocessing: do seperate z......')
                prob_map_interp_xy = torch.zeros(
                    list(_prob_map.size()[:2]) + [_prob_map.size()[2], ] + list(sitk_raw.GetSize()[::-1][1:]), dtype=torch.half)

                for i in range(_prob_map.size(2)):
                    prob_map_interp_xy[:, :, i] = F.interpolate(_prob_map[:, :, i].cuda().float(),
                                                                size=sitk_raw.GetSize()[::-1][1:],
                                                                mode="bilinear").detach().half().cpu()
                del _prob_map

                prob_map_interp = np.zeros(list(prob_map_interp_xy.size()[:2]) + list(sitk_raw.GetSize()[::-1]),
                                        dtype=np.float16)

                for i in range(prob_map_interp.shape[1]):
                    prob_map_interp[:, i] = F.interpolate(prob_map_interp_xy[:, i:i + 1].cuda().float(),
                                                        size=sitk_raw.GetSize()[::-1],
                                                        mode="nearest").detach().half().cpu().numpy()
                del prob_map_interp_xy

            else:
                prob_map_interp = np.zeros(list(_prob_map.size()[:2]) + list(sitk_raw.GetSize()[::-1]), dtype=np.float16)
                for i in range(_prob_map.size(1)):
                    prob_map_interp[:, i] = F.interpolate(_prob_map[:, i:i + 1].cuda().float(),
                                                        size=sitk_raw.GetSize()[::-1],
                                                        mode="trilinear").detach().half().cpu().numpy()
                del _prob_map

            if save_path is None:
                save_dir = rawf.replace(".nii.gz", "_pred.nii.gz")
                assert 1==2
            else:
                uid = rawf.split("/")[-1].replace(".nii.gz", "_pred.nii.gz")
                _save_path = os.path.join(save_path, 'ds_'+str(idx))
                os.makedirs(_save_path, exist_ok=True)
                save_dir = os.path.join(_save_path, uid)
            if args.save_npz: # added 
                save_npz_dir = save_path.replace("nnUNet_inference", "nnUNet_inference_npz")
                save_npz_dir = os.path.join(save_npz_dir, 'ds_'+str(idx))
                os.makedirs(save_npz_dir, exist_ok=True)
                npz_name = rawf.split("/")[-1].replace(".nii.gz", ".npz")
                np.savez_compressed(os.path.join(save_npz_dir, npz_name), softmax=prob_map_interp.squeeze(0))
            
            segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)
            del prob_map_interp
            pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
            pred_sitk.CopyInformation(sitk_raw)
            pred_sitk = resample_image_to_ref(pred_sitk, sitk_raw)
            sitk.WriteImage(pred_sitk, save_dir)
        return

    if pad_flag:
        prob_map = prob_map[:, :,
                   pad_left[0]: img_arr.shape[0] - pad_right[0],
                   pad_left[1]: img_arr.shape[1] - pad_right[1],
                   pad_left[2]: img_arr.shape[2] - pad_right[2]]
    del img_arr

    if (get_do_separate_z(origin_spacing) or get_do_separate_z(current_spacing[::-1])):
        print('postpreprocessing: do seperate z......')
        prob_map_interp_xy = torch.zeros(
            list(prob_map.size()[:2]) + [prob_map.size()[2], ] + list(sitk_raw.GetSize()[::-1][1:]), dtype=torch.half)

        for i in range(prob_map.size(2)):
            prob_map_interp_xy[:, :, i] = F.interpolate(prob_map[:, :, i].cuda().float(),
                                                        size=sitk_raw.GetSize()[::-1][1:],
                                                        mode="bilinear").detach().half().cpu()
        del prob_map

        prob_map_interp = np.zeros(list(prob_map_interp_xy.size()[:2]) + list(sitk_raw.GetSize()[::-1]),
                                   dtype=np.float16)

        for i in range(prob_map_interp.shape[1]):
            prob_map_interp[:, i] = F.interpolate(prob_map_interp_xy[:, i:i + 1].cuda().float(),
                                                  size=sitk_raw.GetSize()[::-1],
                                                  mode="nearest").detach().half().cpu().numpy()
        del prob_map_interp_xy

    else:
        prob_map_interp = np.zeros(list(prob_map.size()[:2]) + list(sitk_raw.GetSize()[::-1]), dtype=np.float16)
        for i in range(prob_map.size(1)):
            prob_map_interp[:, i] = F.interpolate(prob_map[:, i:i + 1].cuda().float(),
                                                  size=sitk_raw.GetSize()[::-1],
                                                  mode="trilinear").detach().half().cpu().numpy()
        del prob_map

    if save_path is None:
        raise ValueError("save_path is None")
        save_dir = rawf.replace(".nii.gz", "_pred.nii.gz")
    else:
        #change name to msd format
        uid = rawf.split("/")[-1].replace('_0000', '')
        save_dir = os.path.join(save_path, uid)

    segmentation = np.argmax(prob_map_interp.squeeze(0), axis=0)


    if args.save_npz:
        # save_npz_dir = save_path.replace("nnUNet_inference", "nnUNet_inference_npz")
        # os.makedirs(save_npz_dir, exist_ok=True)
        # npz_name = uid.replace(".nii.gz", ".npz")
        # np.savez_compressed(os.path.join(save_npz_dir, npz_name), softmax=prob_map_interp.squeeze(0))



        segmentation_final = segmentation
        probabilities_final = prob_map_interp.squeeze(0)

        # NOTE: STEP 0 ==> extract only pancreas and tumor
        pancreas_and_tumor_labels = [13,14,23,24,25]
        segmentation_final_pancreas_and_tumor = np.zeros_like(segmentation_final)
        largest_cc_pred_mask_pancreas_and_tumor = keep_largest_component_multi_cls(segmentation_final, pancreas_and_tumor_labels)
        segmentation_final_pancreas_and_tumor[largest_cc_pred_mask_pancreas_and_tumor==1] = segmentation_final[largest_cc_pred_mask_pancreas_and_tumor==1]
        segmentation_final = segmentation_final_pancreas_and_tumor


        # NOTE: STEP 1 ==> extract the probabilities without saving the entire file!
        bdmap_id = uid.split('.')[0]
        new_row = parse_prediction_new_row(segmentation_final, probabilities_final, bdmap_id, rai_size, origin_spacing)
        
        writeable_new_row_wo_bdmap_id = tuple(map(lambda x: str(x) if isinstance(x, list) or isinstance(x, tuple) else x, new_row[1:])) # turn list and tuple to str
        lock_path = output_csv_path + ".lock"   # File Lock to prevent multi-processing problem
        with thread_lock:   # all threads shares the same threading lock
            with FileLockForAllUsers(lock_path, timeout=60):    # prevent `.lock` permission error when using multiple users
                result_csv_df = pd.read_csv(output_csv_path)
                row_index = result_csv_df["bdmap_id"].tolist().index(bdmap_id)
                result_csv_df.iloc[row_index, 1:] = writeable_new_row_wo_bdmap_id
                result_csv_df.to_csv(output_csv_path, index=False)


    del prob_map_interp
    pred_sitk = sitk.GetImageFromArray(segmentation.astype(np.uint8))
    pred_sitk.CopyInformation(sitk_raw)
    pred_sitk = resample_image_to_ref(pred_sitk, sitk_raw)

    sitk.WriteImage(pred_sitk, save_dir)

if not args.raw_data_dir:
    raw_data_dir = os.getenv('nnUNet_raw_data_base')+"/nnUNet_raw_data/"+task+'/' + args.raw_data_folder
else:
    raw_data_dir = args.raw_data_dir

if not args.save_folder:
    task_name = task.replace('Task0', 'Task')
    args.save_folder = os.getenv('nnUNet_raw_data_base')+"/nnUNet_inference/"+fold_name+'/'+task_name+'/'+os.path.basename(args.raw_data_dir)
print(args.save_folder)
os.makedirs(args.save_folder, exist_ok=True)

# rawf = sorted(glob(raw_data_dir+"/*.nii.gz"))
# if val_keys is not None:
#     valid_rawf = [i for i in rawf if os.path.basename(i).replace('.nii.gz', '').replace('_0000', '') in val_keys]
# else:
#     valid_rawf = rawf

# breakpoint()
if not os.path.exists(args.output_csv):
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    addon_column_names = pd.read_csv("csv_header.csv").columns.tolist()[1:] # the first one is `bdmap_id`, already exists
    copy_df = pd.read_csv(args.input_csv)[["BDMAP ID"]]
    copy_df.rename(columns={"BDMAP ID": "bdmap_id"}, inplace=True)
    for column_name in addon_column_names:
        if "gt" not in column_name: # NOTE: we deal with gt later after inference process!
            copy_df[column_name] = None
    copy_df.to_csv(args.output_csv, index=False)
    os.chmod(args.output_csv, 0o777)    # gives everyone permission to prevent multi-user permission issue. 

# NOTE: use csv for parsing input paths, and insure exact same order!!!!!!!!
input_csv = pd.read_csv(args.input_csv)
output_csv = pd.read_csv(args.output_csv)
output_already_csv = output_csv["bdmap_id"][~pd.isna(output_csv["shape"])].tolist()
input_ct_bdmap_id_list = input_csv["BDMAP ID"].tolist()
valid_rawf= list(set(input_ct_bdmap_id_list) - set(output_already_csv))
# breakpoint()
print(f"\033[31m[FlagshipToCSV][INFO] The current part of job has {len(input_ct_bdmap_id_list)} cases in total. But with respect to the {args.output_csv}, we only work on {len(valid_rawf)}.\033[0m")
valid_rawf=[os.path.join(raw_data_dir, i+'_0000.nii.gz') for i in valid_rawf]

for i, rawf in enumerate(tqdm(valid_rawf)):
    if task.find('500') != -1 or  task.find('005') != -1  or task.find('001')  != -1:
        Inference3D_multiphase(rawf, args.save_folder)
    else:
        Inference3D(rawf, args.save_folder, args.output_csv)
    
    if args.num_examples is not None and i >= args.num_examples:
        break

print('inference done!')
