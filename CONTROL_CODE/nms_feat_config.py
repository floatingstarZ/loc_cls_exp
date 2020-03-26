from commonlibs.common_tools.dir_tools import mkdir
pass_test = False
pass_coco_eval = False
img_meta_img_id = True
use_bayesian_nms = False
use_soft_nms = False
fix_nms_pre_1000 = True
suffix = ''
if use_bayesian_nms:
    suffix = '_Baye_nms' + suffix
if use_soft_nms:
    suffix = '_soft_nms' + suffix
else:
    suffix = '_Normal_nms' + suffix
if fix_nms_pre_1000:
    print('FIX 1000')
# nms feature save
root = '/home/huangziyue/Projects/mmd_test/intermediate_results'

# net_name = 'nms_Retina_r50'
# net_name = 'nms_Cascade_r50'
# net_name = 'nms_Faster_r50'
# net_name = 'nms_Fovea_r50'
# net_name = 'nms_Retina_r101'
# net_name = 'nms_Cascade_r101'
# net_name = 'nms_Faster_r101'
# net_name = 'nms_Fovea_r101'
# net_name = 'nms_Retina_add_ar5_r50'
# net_name = 'nms_Retina_double_r50'
# net_name = 'nms_Free_anchor_r50'
net_name = 'nms_Free_anchor_r101'


int_root = root + '/' + net_name
mkdir(int_root)

save_nms_pre_after = True
nms_root = int_root + '/' + 'NMS_feat' + suffix
mkdir(nms_root)


# print local variable
local_var_dict = locals().copy()
print('################ VARS IN NMS FEAT CONFIG ############################')
for name, v in local_var_dict.items():
    if name[0] == '_':
        continue
    if isinstance(v, bool) or isinstance(v, str):
        print(name, ': ', v)
print('################ VARS IN NMS FEAT CONFIG ############################')




