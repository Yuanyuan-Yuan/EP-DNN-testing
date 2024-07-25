import os
import time
import copy
import argparse
import subprocess
from tqdm import tqdm
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import MODEL_ZOO
from models import build_generator
from utils.misc import bool_parser
from utils.visualizer import HtmlPageVisualizer
from utils.visualizer import postprocess_image
from utils.visualizer import save_image

from jacobian import Jacobian
from jacobian import RobustPCA


def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Synthesize images with pre-trained models.')
    group = parser.add_argument_group('Direction')
    group.add_argument('model_name', type=str,
                        help='Name to the pre-trained model.')
    # group.add_argument('boundary_path', type=str,
    #                    help='Path to the attribute vectors.')
    group.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save the results. If not specified, '
                             'the results will be saved to '
                             '`work_dirs/synthesis/` by default. '
                             '(default: %(default)s)')
    group.add_argument('--num', type=int, default=10,
                        help='Number of samples to synthesize. '
                             '(default: %(default)s)')
    group.add_argument('--batch_size', type=int, default=1,
                        help='Batch size. (default: %(default)s)')
    group.add_argument('--generate_html', type=bool_parser, default=True,
                        help='Whether to use HTML page to visualize the '
                             'synthesized results. (default: %(default)s)')
    group.add_argument('--save_raw_synthesis', type=bool_parser, default=False,
                        help='Whether to save raw synthesis. '
                             '(default: %(default)s)')
    group.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    group.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    group.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    group.add_argument('--randomize_noise', type=bool_parser, default=False,
                        help='Whether to randomize the layer-wise noise. This '
                             'is particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    group.add_argument('--lamb', type=float, default=60)
    # group = parser.add_argument_group('Manipulation')
    # group.add_argument('--vis_size', type=int, default=256,
    #                    help='Size of the visualize images. (default: 256)')
    # group.add_argument('--mani_layers', type=str, default='4,5,6,7',
    #                    help='The layers will be manipulated.'
    #                         '(default: 4,5,6,7). For the eyebrow and lipstick,'
    #                         'using [8-11] layers instead.')
    # group.add_argument('--step', type=int, default=7,
    #                    help='Number of manipulation steps. (default: 7)')
    # group.add_argument('--start', type=int, default=0,
    #                    help='The start index of the manipulation directions.')
    # group.add_argument('--end', type=int, default=1,
    #                    help='The end index of the manipulation directions.')
    # group.add_argument('--start_distance', type=float, default=-30.0,
    #                    help='Start distance for manipulation. (default: -10.0)')
    # group.add_argument('--end_distance', type=float, default=30.0,
    #                    help='End distance for manipulation. (default: 10.0)')
    return parser.parse_args()

args = parse_args()

def syn2jaco(w):
    # wp = w.unsqueeze(1).repeat((1, generator.num_layers, 1))
    wp = generator.truncation(w, args.trunc_psi, args.trunc_layers)
    image = generator.synthesis(wp)['image']
    # images = postprocess_image(images.detach().cpu().numpy())
    image_size = 128
    if image.shape[-1] > image_size:
        scale = image_size / image.shape[-1]
        image = F.interpolate(image, scale_factor=scale)
        image = torch.sum(image, dim=1)
    return image

def syn2image(w):
    # image = generator(z, **synthesis_kwargs)['image']
    with torch.no_grad():
        # wp = w.unsqueeze(1).repeat((1, generator.num_layers, 1))
        wp = generator.truncation(w, args.trunc_psi, args.trunc_layers)
        image = generator.synthesis(wp)['image']
    # images = postprocess_image(images.detach().cpu().numpy())
    image_size = 128
    if image.shape[-1] > image_size:
        scale = image_size / image.shape[-1]
        image = F.interpolate(image, scale_factor=scale)
    
    image = postprocess_image(image.detach().cpu().numpy())
    if len(image.shape) == 4:
        image = image[-1]
    return image

def jaco2mask(jaco, t=None):
    jaco -= jaco.min()
    jaco /= jaco.max()
    jaco *= 255
    jaco = jaco.astype(np.uint8)
    blur = cv2.GaussianBlur(jaco, (5, 5), 0)
    if t is None:
        _, th = cv2.threshold(blur, blur.mean(), 255, cv2.THRESH_BINARY)
    else:
        _, th = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY)
    
    # closing = cv2.morphologyEx(th, cv2.MORPH_CLOSE, 
    #                            np.ones((5, 5), np.uint8)
    #         )
    
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN,
                               np.ones((9, 9), np.uint8)
            )
    
    return blur, th, opening

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0
    return labeled_img

def load_TF(path):
    result_list = np.load(path, allow_pickle=True)
    TF_list = []
    for result in result_list:
        for i in range(len(result['direction_list'])):
            direction = result['direction_list'][i]['direction']
            TF_list += np.vsplit(direction, len(direction))
    print('Load %d TFs.' % len(TF_list))
    return TF_list

def load_unique_TF(path):
    TF = np.load(path)
    print('Loaded %d unique TFs.' % len(TF))
    return TF

def save_unique_TF(in_path, th=0.6):
    out_file = ('unique_%s_' % th) + in_path.split('/')[-1].split('.')[0]
    out_path = '/'.join(in_path.split('/')[:-1]) + '/' + out_file + '.npy'
    TF_list = load_TF(in_path)
    deduplicated = deduplicate_TF(TF_list, th)
    TF = np.stack(deduplicated, 0)
    np.save(out_path, TF)
    print('Saved %d unique TFs in %s.' % (TF.shape[0], out_path))

def vec_norm(v):
    return np.sqrt((v ** 2).sum())

def cosine_similarity(v1, v2):
    return (v1 * v2).sum() / (vec_norm(v1) * vec_norm(v2))

def deduplicate_TF(TF_list, th):
    print('deduplicating TFs using cosine similarity...')
    out_list = []
    for i in tqdm(range(len(TF_list))):
        flag = True
        for j in range(i+1, len(TF_list)):
            sim = cosine_similarity(TF_list[i], TF_list[j])
            if sim > th:
                flag = False
                break
        if flag:
            out_list.append(TF_list[i])
    print('Total %d unique TFs.' % len(out_list))
    return out_list


if __name__ == '__main__':
    mode = 1
    if mode == 1:
        assert args.batch_size == 1

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Parse model configuration.
        if args.model_name not in MODEL_ZOO:
            raise SystemExit(f'Model `{args.model_name}` is not registered in '
                             f'`models/model_zoo.py`!')
        model_config = MODEL_ZOO[args.model_name].copy()
        url = model_config.pop('url')  # URL to download model if needed.

        # Get work directory and job name.
        if args.save_dir:
            work_dir = args.save_dir
        else:
            work_dir = os.path.join('work_dirs', 'synthesis')
        os.makedirs(work_dir, exist_ok=True)
        job_name = f'{args.model_name}_{args.num}'
        if args.save_raw_synthesis:
            os.makedirs(os.path.join(work_dir, job_name), exist_ok=True)

        # Build generation and get synthesis kwargs.
        print(f'Building generator for model `{args.model_name}` ...')
        generator = build_generator(**model_config)
        synthesis_kwargs = dict(trunc_psi=args.trunc_psi,
                                trunc_layers=args.trunc_layers,
                                randomize_noise=args.randomize_noise)
        print(f'Finish building generator.')

        # Load pre-trained weights.
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = os.path.join('checkpoints', args.model_name + '.pth')
        print(f'Loading checkpoint from `{checkpoint_path}` ...')
        if not os.path.exists(checkpoint_path):
            print(f'  Downloading checkpoint from `{url}` ...')
            subprocess.call(['wget', '--quiet', '-O', checkpoint_path, url])
            print(f'  Finish downloading checkpoint.')
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'generator_smooth' in checkpoint:
            generator.load_state_dict(checkpoint['generator_smooth'])
        else:
            generator.load_state_dict(checkpoint['generator'])
        generator = generator.cuda()
        generator.eval()
        print(f'Finish loading checkpoint.')

        cnt = 0
        result_list = []
        for i in tqdm(range(args.num)):
            code = torch.randn(args.batch_size, generator.z_space_dim).cuda()
            with torch.no_grad():
                w = generator.mapping(code, None)['w']
            np_jacobians = Jacobian(syn2jaco, w)
            Js = copy.deepcopy(np_jacobians)
            blur, th, opening = jaco2mask(np.mean(np_jacobians, -1)[0,:,:,0], t=None)
            num_labels, labels_im = cv2.connectedComponents(opening)
            image = syn2image(w)
            labeled = imshow_components(labels_im)

            result = {
                'model_name': args.model_name,
                'seed': image,
                'blur': blur,
                'th': th,
                'opening': opening,
                'labeled': labeled,
                'direction_list': []
            }
            for region_ind in range(num_labels):
                foreground_ind = np.where(labels_im == region_ind)
                background_ind = np.where(labels_im != region_ind)
                w_dim = Js.shape[-1]
                MAX_ITER = 5000
                NUM_RELAX = 0
                NUM_SHRINK = 0
                LAMB = 60
                assert Js.shape[0] == 1
                J = Js[0]
                if len(J.shape) == 4:  # [H, W, 1, latent_dim]
                    Jaco_fore = J[foreground_ind[0], foreground_ind[1], 0]
                    Jaco_back = J[background_ind[0], background_ind[1], 0]
                elif len(J.shape) == 5:  # [channel, H, W, 1, latent_dim]
                    Jaco_fore = J[:, foreground_ind[0], foreground_ind[1], 0]
                    Jaco_back = J[:, background_ind[0], background_ind[1], 0]
                else:
                    raise ValueError(f'Shape of Jacobian is not correct!')
                
                Jaco_fore = np.reshape(Jaco_fore, [-1, w_dim])
                Jaco_back = np.reshape(Jaco_back, [-1, w_dim])

                if Jaco_fore.shape[0] == 0 or Jaco_back.shape[0] == 0:
                    continue

                coef_f = 1 / Jaco_fore.shape[0]
                coef_b = 1 / Jaco_back.shape[0]
                M_fore = coef_f * Jaco_fore.T.dot(Jaco_fore)
                B_back = coef_b * Jaco_back.T.dot(Jaco_back)
                
                # low-rank factorization on foreground
                RPCA = RobustPCA(M_fore, lamb=1/LAMB)
                L_f, _ = RPCA.fit(max_iter=MAX_ITER)
                rank_f = np.linalg.matrix_rank(L_f)

                # low-rank factorization on background
                RPCA = RobustPCA(B_back, lamb=1/LAMB)
                L_b, _ = RPCA.fit(max_iter=MAX_ITER)
                rank_b = np.linalg.matrix_rank(L_b)
                
                # SVD on the low-rank matrix
                _, _, VHf = np.linalg.svd(L_f)
                _, _, VHb = np.linalg.svd(L_b)
                
                F_principal = VHf[:max(1, rank_f - NUM_SHRINK)]  # Principal space of foreground
                relax_subspace = min(max(1, rank_b + NUM_SHRINK), w_dim-1)
                B_null = VHb[relax_subspace:].T  # Null space of background

                F_principal_proj = B_null.dot(B_null.T).dot(F_principal.T)  # Projection
                F_principal_proj = F_principal_proj.T
                
                F_principal_proj /= np.linalg.norm(
                    F_principal_proj, axis=1, keepdims=True)
                #print('F_principal_proj: ', F_principal_proj.shape)
                assert rank_f == F_principal_proj.shape[0]

                direction_item = {
                    'direction': F_principal_proj,
                    'mutated_list': []
                }
                cnt += rank_f
                length = 3
                for vec_ind in range(rank_f):
                    w_hat = w + torch.from_numpy(F_principal_proj[vec_ind] * length).cuda().type(code.dtype)
                    mutated = syn2image(w_hat)
                    direction_item['mutated_list'].append(mutated)

                result['direction_list'].append(direction_item)
            result_list.append(result)
            
            if i and i % 10 == 0:
                saved_path = ('./generated_direction/%s.npy' % args.model_name)
                np.save(saved_path, result_list)
                print('generate total %d directions.' % cnt)

        saved_path = ('./generated_direction/%s.npy' % args.model_name)
        np.save(saved_path, result_list)
        print('generate total %d directions.' % cnt)
    elif mode == 2:
        path = ('./generated_direction/%s.npy' % args.model_name)
        save_unique_TF(path, 0.5)
    elif mode == 3:
        load_unique_TF('./generated_direction/unique_0.5_%s.npy' % args.model_name)
