import copy
from tqdm import tqdm
from random import shuffle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def get_pred(model, x):
    with torch.no_grad():
        if model.exp_name == 'face':
            pred = model(x)[1].argmax(-1)
        else:
            pred = model(x).argmax(-1)
    return pred


def save_model(model, name):
    state = {
        'model': model.state_dict()
    }
    PATH = './repaired_models/%s.pt' % name
    torch.save(state, PATH)
    print('Repaired %s saved in %s.' % (name, PATH))


def compute_adv_loss(model, seed_corpus, T_list, N=5, requires_grad=False):
    ce = nn.CrossEntropyLoss().cuda()
    condition = torch.enable_grad if requires_grad else torch.no_grad
    with condition():
        loss_list = []
        for T in tqdm(T_list):
            loss = 0
            for seed in seed_corpus:
                # pred = model(seed['x'].cuda()).argmax(-1)
                for idx in range(N):
                    x_, z_ = T.mutate(seed)
                    if model.exp_name in ['face']:
                        _, out = model(x_.cuda())
                    else:
                        out = model(x_.cuda())
                    loss += ce(out, seed['y'].cuda())
            loss /= (N * len(seed_corpus))
            loss_list.append(loss if requires_grad else loss.item())
        return loss_list


def result_for_train(result_list):
    ETI_list = []
    label_list = []
    T_idx_list = []
    for result in result_list:
        x, x_ = result['x'], result['x_']
        label = result['y']
        ETI_list.append(x_)
        label_list.append(label)
        T_idx_list.append(result['T'])
    return torch.cat(ETI_list, 0), torch.cat(label_list, 0), torch.LongTensor(T_idx_list)


def repair_model_ETI(model, ETI_tensor, label_tensor, **kwargs):
    assert len(ETI_tensor) == len(label_tensor)
    batch_size = kwargs['batch_size']
    optimizer = torch.optim.SGD(model.parameters(), lr=kwargs['lr'])
    model.train()
    ce = nn.CrossEntropyLoss().cuda()
    index = list(range(len(ETI_tensor)))
    for i in range(kwargs['epoch']):
        shuffle(index)
        index_tensor = torch.from_numpy(np.array(index).astype(np.int64))
        loss_list = []
        for j in tqdm(range(len(ETI_tensor) // batch_size)):
            idx = index_tensor[j * batch_size : (j + 1) * batch_size]
            ETI_input = ETI_tensor[idx]
            label_input = label_tensor[idx]
            ETI_input = ETI_input.cuda()
            label_input = label_input.cuda()
            if model.exp_name in ['face']:
                _, output = model(ETI_input)
            else:
                output = model(ETI_input)
            loss = ce(output, label_input)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
        print('Epoch: %d, Loss: %f' % (i, np.mean(loss_list)))
    model.eval()
    return model
    # save_model(model)

def repair_model_T_aware(model, ETI_tensor, label_tensor, T_idx_tensor, num_T, **kwargs):
    assert len(ETI_tensor) == len(label_tensor)
    batch_size = kwargs['batch_size']
    optimizer = torch.optim.SGD(model.parameters(), lr=kwargs['lr'])
    model.train()
    ce = nn.CrossEntropyLoss().cuda()
    index = list(range(len(ETI_tensor)))
    for i in range(kwargs['epoch']):
        shuffle(index)
        index_tensor = torch.from_numpy(np.array(index).astype(np.int64))
        loss_list = []
        for j in tqdm(range(len(ETI_tensor) // batch_size)):
            idx = index_tensor[j * batch_size : (j + 1) * batch_size]
            ETI_input = ETI_tensor[idx]
            label_input = label_tensor[idx]
            ETI_input = ETI_input.cuda()
            label_input = label_input.cuda()

            if model.exp_name in ['face']:
                _, output = model(ETI_input)
            else:
                output = model(ETI_input)
            
            T_idx = T_idx_tensor[idx]
            total_loss = 0
            for k in range(num_T):
                is_k = (T_idx == k)
                if is_k.sum() == 0:
                    continue
                output_index = is_k.nonzero()[0]
                loss = ce(output[output_index], label_input[output_index])
                loss /= len(output_index)
                total_loss += loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            loss_list.append(total_loss.item())
        print('Epoch: %d, Loss: %f' % (i, np.mean(loss_list)))
    model.eval()
    return model
    # save_model(model)


def eval_model_EP(model, seed_corpus, T_list, N=10):
    # compute the attack success rate for each T
    with torch.no_grad():
        cnt_list = []
        for T in tqdm(T_list):
            cnt = 0
            for seed in seed_corpus:
                pred = get_pred(model, seed['x'].cuda())
                for idx in range(N):
                    x_, z_ = T.mutate(seed)
                    pred_ = get_pred(model, x_.cuda())
                    if pred.item() != pred_.item():
                        cnt += 1
            cnt_list.append(cnt / (N * len(seed_corpus)))
        return cnt_list


def compute_ME_ratio(model, seed_corpus, T_list, idx_list, f=None, **kwargs):
    prev_loss_list = compute_adv_loss(model, seed_corpus, T_list, requires_grad=False)
    for idx in idx_list:
        prev_model = copy.deepcopy(model)

        optimizer = torch.optim.SGD(prev_model.parameters(), lr=kwargs['lr'])
        prev_model.train()
        for i in range(kwargs['epoch']):
            runtime_loss_list = compute_adv_loss(prev_model, seed_corpus, [T_list[idx]], requires_grad=True)
            loss_with_grad = runtime_loss_list[0]
            optimizer.zero_grad()
            loss_with_grad.backward()
            optimizer.step()
        prev_model.eval()

        curr_loss_list = compute_adv_loss(prev_model, seed_corpus, T_list, requires_grad=False)
        increase = (np.array(curr_loss_list) > np.array(prev_loss_list)).astype(np.float32)

        print('%f are mutually exclusive with %d' % (increase.sum() / len(increase), idx))
        print(np.where(increase > 0)[0])
        if f is not None:
            print('%f are mutually exclusive with %d' % (increase.sum() / len(increase), idx), file=f)
            print(np.where(increase > 0)[0], file=f)

if __name__ == '__main__':
    import os
    import argparse
    from tqdm import tqdm
    import torchvision

    import nc_tool

    from synthesize import import_generator

    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('fuzz')
    # group.add_argument('--G_name', type=str, default='stylegan2-ada_afhq512', choices=[
    #     'stylegan2-ada_afhq512', 'stylegan_ffhq256', 'stylegan_car512', 'BigGAN'
    # ])
    group.add_argument('--model', type=str, default='efficientnet_b0', choices=[
        'FaceNet', 'alexnet', 'efficientnet_b0',
        'resnet34', 'vgg16_bn', 'densenet121', 'mobilenet_v2', 'inception_v3',
    ])
    # group.add_argument('--criterion', type=str, default='NC', choices=[
    #     'NC', 'LSC', 'TKNC', 'NLC',
    #     'Entropy', 'Rand', 'KL', 'JS',
    #     # 'WD', 'HD'
    # ])
    group.add_argument('--seed', type=int, default=1234)
    group.add_argument('--num_corpus', type=int, default=20)
    group.add_argument('--num_percep', type=int, default=20)
    group.add_argument('--nc', type=int, default=3)
    # group.add_argument('--image_size', type=int, default=128)
    group.add_argument('--num_class', type=int, default=1000)

    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--num_workers', type=int, default=4)

    group.add_argument('--max_extent', type=float, default=3)
    group.add_argument('--init_p', type=int, default=10)
    # group.add_argument('--trunc_psi', type=float, default=0.7)
    # group.add_argument('--trunc_layers', type=int, default=8)
    # group.add_argument('--reg_threshold', type=float, default=0.5)

    args = parser.parse_args()

    assert args.batch_size == 1

    if args.model in ['alexnet']:
        args.G_name = 'stylegan2-ada_afhq512'
    elif args.model in ['FaceNet']:
        args.G_name = 'stylegan_ffhq256'
    elif args.model in ['efficientnet_b0']:
        args.G_name = 'stylegan_car512'
    else:
        args.G_name = 'BigGAN'

    from prepare_T import GtoParams
    args.image_size = GtoParams[args.G_name]['image_size']
    if 'stylegan' in args.G_name:
        args.trunc_psi = GtoParams[args.G_name]['trunc_psi']
        args.trunc_layers = GtoParams[args.G_name]['trunc_layers']

    from prepare_model import get_model
    model = get_model(args.model)
    model = model.cuda()

    if 'stylegan' in args.G_name:
        G = import_generator(args.G_name)
    elif args.G_name == 'BigGAN':
        import sys
        sys.path.insert(0, './BigGAN/')
        from import_biggan import get_G
        G = get_G()
        Z_DIM = 120
        LABEL_INDEX = 0

    def to_input(image, image_size=args.image_size):
        with torch.no_grad():
            scale = args.image_size / image.shape[-1]
            input = F.interpolate(image, scale_factor=scale)
            return input

    print('Building seed corpus...')
    seed_corpus = []
    with torch.no_grad():    
        for i in tqdm(range(args.num_corpus)):
            if 'stylegan' in args.G_name:
                code = torch.randn(args.batch_size, G.z_space_dim).cuda()
                z = G.mapping(code, None)['w']
                wp = G.truncation(z, args.trunc_psi, args.trunc_layers)
                image = G.synthesis(wp)['image']
            else:
                z = torch.randn(args.batch_size, Z_DIM).cuda()
                y = (torch.ones((args.batch_size, )) * LABEL_INDEX).type(torch.LongTensor).cuda()
                y_embed = G.shared(y)
                image = G((z, y_embed))
            x = to_input(image)
            if args.model in ['FaceNet']:
                y = torch.zeros((args.batch_size,)).type(torch.LongTensor)
            else:
                y = model(x).argmax(-1)
            seed_corpus.append({
                'x': x.cpu(), 'z': z.cpu(), 'y': y.cpu(),
                'p': args.init_p
            })

    from transformation import *
    basic_T_list = [
        Noise(), Brightness(), Contrast(), Blur(),
        Translation(), Scale(), Rotation(), Shear(), Reflection(),
        Cloud(), Fog(), Snow(), Rain(),
    ]
    # 7: Shear()

    from prepare_T import get_stylize, get_perceptual, get_class_perceptual
    stylize_list = get_stylize(args.image_size)
    if 'stylegan' in args.G_name:
        perceptual_list = get_perceptual(args.G_name)
    else:
        perceptual_list = get_class_perceptual(args.G_name, LABEL_INDEX, args.num_percep)

    print('%d basic T.\n%d stylize T.\n%d perceptual T.' \
        % (len(basic_T_list), len(stylize_list), len(perceptual_list))
        )

    full_T_list = basic_T_list + stylize_list
    if args.num_percep > 0:
        full_T_list += perceptual_list[:args.num_percep]
    else:
        full_T_list += perceptual_list
    print('Total %d transformations.' % len(full_T_list))

    # get all result/seed
    prefix = './exp_result/'
    name_list = []
    for name in os.listdir(prefix):
        if (args.model in name) and ('ARand' not in name) and (int(name.split('-')[2]) == args.num_percep):
            name_list.append(name)

    result_list = []
    EP_counter = torch.zeros(len(full_T_list))
    T_counter = torch.zeros(len(full_T_list))
    for name in name_list:
        result_list += torch.load(prefix + name)['result']
        EP_counter += torch.load(prefix + name)['EP_counter']
        T_counter += torch.load(prefix + name)['T_counter']

    print('Total %d ETI.' % len(result_list))

    # compare different repairing schemes
    ETI_tensor, label_tensor, T_idx_tensor = result_for_train(result_list)
    T_cnt_dict = {}
    for i in range(len(full_T_list)):
        cnt = (T_idx_tensor == i).sum().item()
        T_cnt_dict[i] = (cnt, cnt / len(ETI_tensor))
    print('ETI_tensor: ', ETI_tensor.size())
    prev_ASR_list = eval_model_EP(model, seed_corpus, full_T_list, N=10)
    
    ETI_repaired_model = repair_model_ETI(copy.deepcopy(model), ETI_tensor, label_tensor, lr=2e-5, epoch=5, batch_size=8)

    T_aware_repaired_model = repair_model_T_aware(copy.deepcopy(model), ETI_tensor, label_tensor, T_idx_tensor, num_T=len(T_counter),
                                                lr=2e-5, epoch=5, batch_size=8)

    curr_ASR_list_ETI = eval_model_EP(ETI_repaired_model, seed_corpus, full_T_list, N=10)
    curr_ASR_list_T_aware = eval_model_EP(T_aware_repaired_model, seed_corpus, full_T_list, N=10)

    print('prev ASR avg: %f,  min: %f, max: %f' % (np.mean(prev_ASR_list), np.min(prev_ASR_list), np.max(prev_ASR_list)))
    print('curr ETI ASR avg: %f, min: %f, max: %f' % (np.mean(curr_ASR_list_ETI), np.min(curr_ASR_list_ETI), np.max(curr_ASR_list_ETI)))
    print('curr T aware ASR avg: %f, min: %f, max: %f' % (np.mean(curr_ASR_list_T_aware), np.min(curr_ASR_list_T_aware), np.max(curr_ASR_list_T_aware)))

    with open('./exp_log_ME/%s.txt' % args.model, 'a') as f:
        print('prev ASR avg: %f,  min: %f, max: %f' % (np.mean(prev_ASR_list), np.min(prev_ASR_list), np.max(prev_ASR_list)), file=f)
        print('curr ETI ASR avg: %f, min: %f, max: %f' % (np.mean(curr_ASR_list_ETI), np.min(curr_ASR_list_ETI), np.max(curr_ASR_list_ETI)), file=f)
        print('curr T aware ASR avg: %f, min: %f, max: %f' % (np.mean(curr_ASR_list_T_aware), np.min(curr_ASR_list_T_aware), np.max(curr_ASR_list_T_aware)), file=f)
    
    np.save('./exp_ASR/%s-prev.npy' % model.exp_name, prev_ASR_list)
    np.save('./exp_ASR/%s-ETI.npy' % model.exp_name, curr_ASR_list_ETI)
    np.save('./exp_ASR/%s-T.npy' % model.exp_name, curr_ASR_list_T_aware)
    
    save_model(ETI_repaired_model, '%s-ETI' % model.exp_name)
    save_model(T_aware_repaired_model, '%s-T_aware' % model.exp_name)
    