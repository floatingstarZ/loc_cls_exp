pass_test = True
evaluate_map = False
evaluate_out_iou_score = True
outiou_score_file = ''
# show local var

local_var_dict = locals().copy()
print('################ VARS IN EVALUATE CONFIG ############################')
for name, v in local_var_dict.items():
    if name[0] == '_':
        continue
    if isinstance(v, bool) or isinstance(v, str):
        print(name, ': ', v)
print('################ VARS IN EVALUATE CONFIG ############################')





