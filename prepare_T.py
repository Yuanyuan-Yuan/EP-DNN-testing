import json
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformation import Stylize
from transformation import Perceptual
from transformation import ClassPerceptual

from style_operator import StylizeModel
from synthesize import import_generator

GtoParams = {
    'stylegan2-ada_afhq512': {
        'max_step': 15 * 2,
        'trunc_psi': 0.7,
        'trunc_layers': 8,
        'image_size': 128,
    },
    'stylegan_ffhq256': {
        'max_step': 3 * 2,
        'trunc_psi': 0.5,
        'trunc_layers': 8,
        'image_size': 64,
    },
    'stylegan_car512': {
        'max_step': 2 * 2,
        'trunc_psi': 0.4,
        'trunc_layers': 8,
        'image_size': 128,
    },
    'BigGAN': {
        'max_step': 5 * 2,
        'image_size': 128,
    },
}

def get_perceptual(G_name):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G = import_generator(G_name)
    G = G.to(device)
    params = GtoParams[G_name]
    
    def trans(w):
        with torch.no_grad():
            wp = G.truncation(w, params['trunc_psi'], params['trunc_layers'])
            image = G.synthesis(wp)['image']
            scale = params['image_size'] / image.shape[-1]
            input = F.interpolate(image, scale_factor=scale)
            return input

    direction_path = './generated_direction/unique_0.5_%s.npy' % G_name
    direction_arr = np.load(direction_path)
    direction_arr = direction_arr.astype(np.float32)
    T_list = []
    for i in range(len(direction_arr)):
        direction = torch.from_numpy(direction_arr[i])
        T_list.append(Perceptual(direction, i, trans, params['max_step']))
    return T_list

def get_class_perceptual(G_name, label_index, num):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import sys
    sys.path.insert(0, './BigGAN/')
    from import_biggan import get_G
    G = get_G()
    G.to(device)
    params = GtoParams[G_name]
    
    def trans(z, y):
        with torch.no_grad():
            y_embed = G.shared(y)
            image = G((z, y_embed))
            scale = params['image_size'] / image.shape[-1]
            input = F.interpolate(image, scale_factor=scale)
            return input

    direction_path = './generated_direction/unique_0.5_%s.npy' % G_name
    direction_arr = np.load(direction_path, allow_pickle=True).item()[label_index]
    if num > 0:
        direction_arr = direction_arr[:num]
    direction_arr = direction_arr.astype(np.float32)
    T_list = []
    for i in range(len(direction_arr)):
        direction = torch.from_numpy(direction_arr[i])
        T_list.append(ClassPerceptual(direction, i, trans, params['max_step']))
    return T_list

def get_stylize(image_size, style_json='./generated_style/style-0.5.json'):
    stylize_model = StylizeModel(image_size, style_json=style_json)
    trans = stylize_model.transform
    with open(style_json, 'r') as f:
        style_path_list = json.load(f)
    T_list = []
    for i, style_path in enumerate(style_path_list):
        T_list.append(Stylize(style_path, i, trans))
    return T_list

def get_unique_style(image_size, threshold):
    stylize_model = StylizeModel(image_size, 2000)
    unique_style_list = stylize_model.deduplicate_style(threshold)
    print('Found %d unique styles.' % len(unique_style_list))
    out_path = './generated_style/style-%s.json' % (threshold)
    with open(out_path, 'w') as f:
        json.dump(unique_style_list, f)
    print('Saved %d unique styles in %s.' % (len(unique_style_list), out_path))


if __name__ == '__main__':
    # get_unique_style(32, 0.5)
    get_class_perceptual('BigGAN', 0)