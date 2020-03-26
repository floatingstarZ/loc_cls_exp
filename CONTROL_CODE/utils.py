import json
import pickle as pkl
import numpy as np
import torch

def tensor2array(tensor):
    return tensor.cpu().detach().numpy()

def pklsave(obj, file_path):
    with open(file_path, 'wb+') as f:
        pkl.dump(obj, f)
        print('SAVE OBJ: %s' % file_path)

def jsonsave(obj, file_path):
    with open(file_path, 'wt+') as f:
        json.dump(obj, f)
        print('SAVE JSON: %s' % file_path)

import os
def mkdir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('Make Dir %s | | @_@')





