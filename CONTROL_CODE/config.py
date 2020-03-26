import os

root = '/home/huangziyue/Projects/MMD_test/intermediate_results'

net_name = 'Retina_r18_new'


fix_decoder = False
fix_head = False  # not only_train_decoder
fix_backbone_neck = False
# 只训练头部的两层。reg_convs，cls_convs为另一部分。
def decoder_filter(model):
    filt_name = []
    for name, value in model.named_parameters():
        if 'decoder' in name:
            filt_name.append(name)
    return filt_name

def head_filter(model):
    filt_name = []
    for name, value in model.named_parameters():
        if 'head' in name or '':
            filt_name.append(name)
    return filt_name

def backbone_head_filter(model):
    filt_name = []
    for name, value in model.named_parameters():
        if 'backbone' in name or 'neck' in name:
            filt_name.append(name)
    return filt_name


load_pretrain = False
pre = '########################  '
aft = '  ########################'
if load_pretrain:
    print(pre + 'LOAD PRETRAINED' + aft)
transform_input = False
if transform_input:
    print(pre + 'TRASFORM DECODER INPUT AS' + aft)
decoder_input_feats = False
if decoder_input_feats:
    print(pre + 'DECODER INPUT FEATS (CENTER NET)'+aft)

head_test = True
decoder_test = False

train_with_nms_loss = False

###################################################
use_specific_args = True
CUDA_VISIBLE_DEVICES = 5

args = ''
if use_specific_args:
    if net_name == 'Retina_r18':
        args = ['./configs/retinanet_r18_toy.py',
                './results/Retina_r18_toy_test/latest.pth',
                '--out', './results/RESULT_Retina_r18_toy.pkl',
                '--eval', 'bbox'
                ]
    elif net_name == 'Retina_r18_new':
        args = ['./configs/retinanet_r18_toy.py',
                './results/Retina_r18_outer/latest.pth',
                '--out', './results/RESULT_Retina_r18_toy.pkl',
                '--eval', 'bbox'
                ]
