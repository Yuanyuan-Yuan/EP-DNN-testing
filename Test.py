import os
import json
import time
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Tester:
    def __init__(self, args, model, seed_corpus, objective, T_list):
        assert args.batch_size == 1
        self.args = args
        self.model = model
        self.seed_corpus = seed_corpus
        self.init_length = len(seed_corpus)
        self.objective = objective
        self.T_list = T_list

        self.result = []
        self.EP_counter = np.zeros(len(self.T_list))
        self.T_counter = np.zeros(len(self.T_list))
        self.EP_cnt_list = []
        
        self.delta_time = 0
        self.epoch = 0        

    def can_terminate(self):
        condition = sum([
            self.get_p_sum() == 0,
            len(np.where(self.EP_counter == 0)[0]) == 0,
            self.epoch >= 10000,
            # self.delta_time > 60 * 10 * 1,
        ]) 
        return condition > 0

    def get_p_sum(self):
        return np.sum([
            seed['p'] for seed in self.seed_corpus
        ])

    def select_input(self):
        prior = [seed['p'] for seed in self.seed_corpus]
        idx = np.random.choice(len(self.seed_corpus), p=prior/np.sum(prior))
        self.seed_corpus[idx]['p'] -= 1
        return self.seed_corpus[idx]

    def select_T(self):
        idx = np.random.choice(len(self.T_list))
        self.T_counter[idx] += 1
        return self.T_list[idx], idx

    def mutate(self, seed):
        T, T_idx = self.select_T()
        x_, z_ = T.mutate(seed)
        return x_, z_, T_idx

    def create_input(self, x, z, y):
        self.seed_corpus.append({
                'x': x.cpu(), 'z': z.cpu(), 'y': y.cpu(),
                'p': self.args.init_p
            })

    def mis_pred(self, x, x_):
        with torch.no_grad():
            pred = self.model(x.cuda()).argmax(-1)
            pred_ = self.model(x_.cuda()).argmax(-1)
            return (pred != pred_).sum(0) > 0

    def gain(self, seed, x_):
        if self.args.criterion in ['Rand', 'ARand']:
            return True
        elif args.criterion in ['Entropy']:
            return self.objective.gain(seed['x'].cuda(), x_.cuda())
        elif args.criterion in ['KL', 'JS', 'WD', 'HD']:
            raise NotImplementedError                
        elif args.criterion in ['LSC']:
            return self.objective.gain((x_.cuda(), seed['y'].cuda()))
        else:
            return self.objective.gain(x_.cuda())

    def run(self):
        assert args.criterion not in ['KL', 'JS', 'WD', 'HD', 'PRand']
        print('Start testing.')
        start_time = time.time()
        while not self.can_terminate():

            self.EP_cnt_list.append(len(np.where(self.EP_counter > 0)[0]))
            
            seed = self.select_input()
            x_, z_, T_idx = self.mutate(seed)

            if self.fault(seed['x'], x_, T_idx):
                self.result.append({
                    'x': seed['x'].cpu(), 'z': seed['z'].cpu(), 'y': seed['y'].cpu(),
                    'x_': x_.cpu(), 'z_': z_.cpu(),
                    'T': T_idx,
                })
                # self.create_input(x_, z_)
                # self.print_info()
            elif self.gain(seed, x_):
                if self.args.criterion not in ['ARand']:
                    self.create_input(x_, z_, seed['y'])

            self.delta_time = time.time() - start_time
            self.epoch += 1
        self.exit()

    def print_info(self):
        print('Epoch: %d' % self.epoch)
        print('Delta time: %fs' % self.delta_time)
        print('%d error-triggering inputs.' % len(self.result))
        print('%d vulnerable properties.' % len(np.where(self.EP_counter > 0)[0]))

    def exit(self):
        self.print_info()

        prefix = './exp_result'
        name_list = os.listdir(prefix)
        saved_name = '%s-%s-%s-%d' % (self.args.mode, self.args.model, self.args.criterion, self.args.num_percep)
        cnt = [1 if saved_name + '-' in name else 0 for name in name_list]
        path = '%s/%s-%d.pt' % (prefix, saved_name, np.sum(cnt))        
        state = {
            'result': self.result,
            'EP_counter': torch.from_numpy(self.EP_counter),
            'T_counter': torch.from_numpy(self.T_counter),
        }
        torch.save(state, path)
        print('Saved results in %s' % path)
        
        log_path = './exp_log/%s-%s.json' % (self.args.mode, self.args.model, self.args.criterion)
        with open(log_path, 'w') as f:
            json.dump(self.EP_cnt_list, f)
        print('Saved log in %s' % log_path)

    def fault(self, x, x_):
        raise NotImplementedError


class PropertyTester(Tester):

    def select_T(self):
        assert len(self.T_list) == len(self.EP_counter)
        prob = np.ones(len(self.T_list))        
        mask = np.where(self.EP_counter > 0)
        prob[mask] = 0
        idx = np.random.choice(len(self.T_list), p=prob/np.sum(prob))
        self.T_counter[idx] += 1
        return self.T_list[idx], idx

    def fault(self, x, x_, T_idx):
        if self.mis_pred(x, x_):
            self.EP_counter[T_idx] += 1
            if self.EP_counter[T_idx] == 1:
                return True
        return False


class InputTester(Tester):

    def fault(self, x, x_, T_idx):
        if self.mis_pred(x, x_):
            self.EP_counter[T_idx] += 1
            return True
        return False

class PInputTester(InputTester):

    def run(self):
        assert args.criterion in ['KL', 'JS', 'WD', 'HD', 'PRand']
        start_time = time.time()
        record_list = []
        priority_list = []
        print('Preparing mutated inputs...')
        for seed in tqdm(self.seed_corpus):
            for _ in range(self.args.init_p * 2):
                x_, z_, T_idx = self.mutate(seed)
                record_list.append((x_, z_, T_idx, seed))
                if args.criterion not in ['PRand']:
                    priority_list.append(-1 * self.objective.compute_div(seed['x'].cuda(), x_.cuda()))
                    # descending order
                else:
                    priority_list.append(0)
        idx_list = np.argsort(priority_list)

        print('Start testing.')
        while not self.can_terminate():

            self.EP_cnt_list.append(self.EP_counter.sum())

            record_idx = idx_list[self.epoch]
            (x_, z_, T_idx, seed) = record_list[record_idx]
            if self.fault(x, x_, T_idx):
                self.result.append({
                    'x': seed['x'].cpu(), 'z': seed['z'].cpu(), 'y': seed['y'].cpu(),
                    'x_': x_.cpu(), 'z_': z_.cpu(),
                    'T': T_idx,
                })
                # self.create_input(x_, z_)
                # self.print_info()

            self.delta_time = time.time() - start_time
            self.epoch += 1
        self.exit()

from face import PairwiseDistance

class RegInputTester(InputTester):
    def __init__(self, args, model, seed_corpus, objective, T_list):
        super(RegInputTester, self).__init__(args, model, seed_corpus, objective, T_list)
        self.l2_dist = PairwiseDistance(2)

    def mis_pred(self, x, x_):
        with torch.no_grad():
            feat, pred = self.model(x.cuda())
            feat_, pred_ = self.model(x_.cuda())
            dist = self.l2_dist.apply(feat, feat_).item()
            return (1 / (1 + dist)) < self.args.reg_threshold

class PRegInputTester(PInputTester):
    def __init__(self, args, model, seed_corpus, objective, T_list):
        super(PRegTester, self).__init__(args, model, seed_corpus, objective, T_list)
        self.l2_dist = PairwiseDistance(2)

    def mis_pred(self, x, x_):
        with torch.no_grad():
            feat, pred = self.model(x.cuda())
            feat_, pred_ = self.model(x_.cuda())
            dist = self.l2_dist.apply(feat, feat_).item()
            return (1 / (1 + dist)) < self.args.reg_threshold

class RegPropertyTester(PropertyTester):
    def __init__(self, args, model, seed_corpus, objective, T_list):
        super(RegPropertyTester, self).__init__(args, model, seed_corpus, objective, T_list)
        self.l2_dist = PairwiseDistance(2)

    def mis_pred(self, x, x_):
        with torch.no_grad():
            feat, pred = self.model(x.cuda())
            feat_, pred_ = self.model(x_.cuda())
            dist = self.l2_dist.apply(feat, feat_).item()
            return (1 / (1 + dist)) < self.args.reg_threshold



if __name__ == '__main__':
    from tqdm import tqdm
    import torchvision

    import nc_tool

    from synthesize import import_generator
    import objectives
    from objectives import HYPER

    import sys
    import signal
    def signal_handler(sig, frame):
        print('You pressed Ctrl+C!')
        try:
            if tester is not None:
                tester.exit()
            else:
                print('No result saved!')
        except:
            pass
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group('fuzz')
    group.add_argument('G_name', type=str, choices=['stylegan2-ada_afhq512', 'stylegan_ffhq256', 'stylegan_car512', 'BigGAN'])
    group.add_argument('--model', type=str, default='alexnet', choices=[
        'FaceNet', 'alexnet', 'efficientnet_b0',
        'resnet34', 'vgg16_bn', 'densenet121', 'mobilenet_v2', 'inception_v3',
    ])
    group.add_argument('--criterion', type=str, default='NC', choices=[
        'NC', 'LSC', 'TKNC', 'NLC',
        'Entropy', 'Rand', 'KL', 'JS',
        'WD', 'HD', 'ARand', 'PRand'
    ])
    group.add_argument('--mode', type=str, default='input', choices=['input', 'property'])

    group.add_argument('--seed', type=int, default=1234)
    group.add_argument('--num_corpus', type=int, default=500)
    group.add_argument('--num_percep', type=int, default=-1)
    group.add_argument('--nc', type=int, default=3)
    # group.add_argument('--image_size', type=int, default=128)
    group.add_argument('--num_class', type=int, default=1000)

    group.add_argument('--batch_size', type=int, default=1)
    group.add_argument('--num_workers', type=int, default=4)

    group.add_argument('--max_extent', type=float, default=3)
    group.add_argument('--init_p', type=int, default=10)
    # group.add_argument('--trunc_psi', type=float, default=0.7)
    # group.add_argument('--trunc_layers', type=int, default=8)
    group.add_argument('--reg_threshold', type=float, default=0.5)

    args = parser.parse_args()

    assert args.batch_size == 1

    from prepare_T import GtoParams
    args.image_size = GtoParams[args.G_name]['image_size']
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

    def to_input(image, image_size=args.image_size):
        with torch.no_grad():
            scale = args.image_size / image.shape[-1]
            input = F.interpolate(image, scale_factor=scale)
            return input

    print('Building seed corpus...')
    seed_corpus = get_seed_corpus(args.model)

    from transformation import *
    basic_T_list = [
        Noise(), Brightness(), Contrast(), Blur(),
        Translation(), Scale(), Rotation(), Shear(), Reflection(),
        Cloud(), Fog(), Snow(), Rain(),
    ]

    from prepare_T import get_stylize, get_perceptual
    stylize_list = get_stylize(args.image_size)
    perceptual_list = get_perceptual(args.G_name)

    print('%d basic T.\n%d stylize T.\n%d perceptual T.' \
        % (len(basic_T_list), len(stylize_list), len(perceptual_list))
        )

    full_T_list = basic_T_list + stylize_list
    if args.num_percep > 0:
        full_T_list += perceptual_list[:args.num_percep]
    else:
        full_T_list += perceptual_list
    print('Total %d transformations.' % len(full_T_list))

    if args.criterion in ['Rand', 'ARand', 'PRand']:
        objective = getattr(objectives, args.criterion)()
    elif args.criterion in ['Entropy']:
        objective = getattr(objectives, args.criterion)(model)
    elif args.criterion in ['KL', 'JS', 'WD', 'HD']:
        objective = getattr(objectives, args.criterion)(model, None)
    else:
        random_input = torch.randn(1, args.nc, args.image_size, args.image_size).cuda()
        layer_size_dict = nc_tool.get_layer_output_sizes(model, random_input)
        hyper = HYPER[args.criterion]
        if args.criterion in ['LSC']:
            objective = getattr(objectives, args.criterion)(model, layer_size_dict, hyper,
                                                            min_var=1e-5, num_class=args.num_class
                                                        )
            # data_tuple_list = [()]
            from prepare_training_data import get_train_loader
            train_loader = get_train_loader(model.exp_name)
            objective.build(train_loader)
        else:
            objective = getattr(objectives, args.criterion)(model, layer_size_dict, hyper)

    if args.mode == 'input':
        if args.criterion in ['KL', 'JS', 'WD', 'HD', 'PRand']:
            Engine = PRegInputTester if args.model in ['FaceNet'] else PInputTester
        else:
            Engine = RegInputTester if args.model in ['FaceNet'] else InputTester
    elif args.mode == 'property':
        Engine = RegPropertyTester if args.model in ['FaceNet'] else PropertyTester

    tester = Engine(args, 
                model=model,
                seed_corpus=seed_corpus,
                objective=objective,
                T_list=full_T_list,
            )
    tester.run()
    print('%f s' % tester.delta_time)