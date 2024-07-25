from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyflann import FLANN
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.stats import wasserstein_distance

import nc_tool as tool

HYPER = {
    'NLC': None,
    'NC': 0.75,
    'LSC': 10,
    'CC': 10,
    'TKNC': 10,
}

class Rand:
    def __init__(self):
        pass
    def gain(self):
        return True

class RandRand:
    def __init__(self):
        pass
    def gain(self):
        return True

class PRand:
    def __init__(self):
        pass
    def gain(self):
        return True

class Entropy:
    def __init__(self, model):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)

    def compute_entropy(self, data):
        out, *_ = self.model(data)
        ent = (-out * torch.log2(out)).sum()
        return ent

    def gain(self, data, new_data):
        return self.compute_entropy(new_data) > self.compute_entropy(data)

class Diversity:
    def __init__(self, model, target_layer_idx=None):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        layer_dict = tool.get_model_layers(self.model)
        if target_layer_idx is not None:
            idx = target_layer_idx
        else:
            idx = len(layer_dict.keys()) // 2
        self.target_layer_dict = {
            'layer': layer_dict[list(layer_dict.keys())[idx]]
        }
        # use dict here to fit the implementation of `get_model_layers`

    def compute_div(self):
        raise NotImplementedError

    def priority(self, x, x_):
        dic = tool.get_layer_output(self.model, x, self.target_layer_dict)
        out = list(dic.items())[0]
        dic_ = tool.get_layer_output(self.model, x_, self.target_layer_dict)
        out_ = list(dic_.items())[0]
        return self.compute_div(out, out_)

def kl_divergence(src, flw):
    kld = (src * torch.log(src / flw)).sum()
    return kld

class KL(Diversity):
    def compute_div(self, src, flw):
        return kl_divergence(src, flw).item()

class JS(Diversity):
    def compute_div(self, src, flw):
        m = (src + flw) / 2
        jsd = (kl_divergence(src, m) + kl_divergence(flw, m)) / 2
        return jsd.item()

class WD(Diversity):
    def compute_div(self, src, flw):
        return wasserstein_distance(
            src.cpu().numpy(), flw.cpu().numpy()
        )

class HD(Diversity):
    def compute_div(self, src, flw):
        hd = (torch.sqrt(
                (src.sqrt() - flw.sqrt()).square().sum()
            )) / np.sqrt(2)
        return hd.item()

class Coverage:
    def __init__(self, model, layer_size_dict, hyper=None, **kwargs):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.model.to(self.device)
        self.layer_size_dict = layer_size_dict
        self.init_variable(hyper, **kwargs)

    def init_variable(self):
        raise NotImplementedError
        
    def calculate(self):
        raise NotImplementedError

    def coverage(self):
        raise NotImplementedError

    def save(self):
        raise NotImplementedError

    def load(self):
        raise NotImplementedError

    def get_cove_dict(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building is not needed.')

    def assess(self, data_loader):
        for data, *_ in tqdm(data_loader):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.step(data)

    def step(self, data):
        cove_dict = self.calculate(data)
        gain = self.gain(cove_dict)
        if gain is not None:
            self.update(cove_dict, gain)

    def update(self, all_cove_dict, delta=None):
        self.coverage_dict = all_cove_dict
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(all_cove_dict)

    # def gain(self, cove_dict_new):
    #     new_rate = self.coverage(cove_dict_new)
    #     return new_rate - self.current

    def gain(self, data):
        cove_dict = self.calculate(data)
        delta = self.coverage(cove_dict) - self.current
        if delta > 0:
            self.update(cove_dict, delta)
            return True
        return False



class SurpriseCoverage(Coverage):
    def init_variable(self, hyper, min_var, num_class):
        self.name = self.get_name()
        assert self.name in ['LSC', 'DSC', 'MDSC']
        assert hyper is not None
        self.threshold = hyper
        self.min_var = min_var
        self.num_class = num_class
        self.data_count = 0
        self.current = 0
        self.coverage_set = set()
        self.mask_index_dict = {}
        self.mean_dict = {}
        self.var_dict = {}
        self.kde_cache = {}
        self.SA_cache = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.mask_index_dict[layer_name] = torch.ones(layer_size[0]).type(torch.LongTensor).to(self.device)
            self.mean_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)
            self.var_dict[layer_name] = torch.zeros(layer_size[0]).to(self.device)

    def get_name(self):
        raise NotImplementedError

    def build(self, data_loader):
        print('Building Mean & Var...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            # print(data.size())
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            self.set_meam_var(data, label)
        self.set_mask()
        print('Building SA...')
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.build_SA(data, label)
        self.to_numpy()
        if self.name == 'LSC':
            self.set_kde()

    def assess(self, data_loader):
        for i, (data, label) in enumerate(tqdm(data_loader)):
            if isinstance(data, tuple):
                data = (data[0].to(self.device), data[1].to(self.device))
            else:
                data = data.to(self.device)
            label = label.to(self.device)
            self.step(data, label)

    def step(self, data, label):
        cove_set = self.calculate(data, label)
        gain = self.gain(cove_set)
        if gain is not None:
            self.update(cove_set, gain)

    def set_meam_var(self, data, label):
        batch_size = label.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            self.data_count += batch_size
            self.mean_dict[layer_name] = ((self.data_count - batch_size) * self.mean_dict[layer_name] + layer_output.sum(0)) / self.data_count
            self.var_dict[layer_name] = (self.data_count - batch_size) * self.var_dict[layer_name] / self.data_count \
            + (self.data_count - batch_size) * ((layer_output - self.mean_dict[layer_name]) ** 2).sum(0) / self.data_count ** 2

    def set_mask(self):
        feature_num = 0
        for layer_name in self.mean_dict.keys():
            self.mask_index_dict[layer_name] = (self.var_dict[layer_name] >= self.min_var).nonzero()
            feature_num += self.mask_index_dict[layer_name].size(0)
        print('feature_num: ', feature_num)

    def build_SA(self, data_batch, label_batch):
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1) # [batch_size, num_neuron]
        # print('SA_batch: ', SA_batch.size())
        SA_batch = SA_batch[~torch.any(SA_batch.isnan(), dim=1)]
        SA_batch = SA_batch[~torch.any(SA_batch.isinf(), dim=1)]
        for i, label in enumerate(label_batch):
            if int(label.cpu()) in self.SA_cache.keys():
                self.SA_cache[int(label.cpu())] += [SA_batch[i].detach().cpu().numpy()]
            else:
                self.SA_cache[int(label.cpu())] = [SA_batch[i].detach().cpu().numpy()]

    def to_numpy(self):
        for k in self.SA_cache.keys():
            self.SA_cache[k] = np.stack(self.SA_cache[k], 0)

    def set_kde(self):
        raise NotImplementedError

    def calculate(self):
        raise NotImplementedError

    def update(self, cove_set, delta=None):
        self.coverage_set = cove_set
        if delta:
            self.current += delta
        else:
            self.current = self.coverage(self.coverage_set)

    def get_cove_dict(self):
        return self.coverage_set

    def coverage(self, cove_set):
        return len(cove_set)

    # def gain(self, cove_set_new):
    #     new_rate = self.coverage(cove_set_new)
    #     return new_rate - self.current

    def save(self, path):
        print('Saving recorded %s in %s...' % (self.name, path))
        state = {
            'coverage_set': list(self.coverage_set),
            'mask_index_dict': self.mask_index_dict,
            'mean_dict': self.mean_dict,
            'var_dict': self.var_dict,
            'SA_cache': self.SA_cache
        }
        torch.save(state, path)

    def load(self, path):
        print('Loading saved %s in %s...' % (self.name, path))
        state = torch.load(path)
        self.coverage_set = set(state['coverage_set'])
        self.mask_index_dict = state['mask_index_dict']
        self.mean_dict = state['mean_dict']
        self.var_dict = state['var_dict']
        self.SA_cache = state['SA_cache']
        loaded_cov = self.coverage(self.coverage_set)
        print('Loaded coverage: %f' % loaded_cov)


class NLC(Coverage):
    def init_variable(self, hyper=None):
        assert hyper is None, 'NLC has no hyper-parameter'
        self.estimator_dict = {}
        self.current = 1
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.estimator_dict[layer_name] = tool.Estimator(feature_num=layer_size[0])
    
    def calculate(self, data):
        stat_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            info_dict = self.estimator_dict[layer_name].calculate(layer_output.to(self.device))
            stat_dict[layer_name] = (info_dict['Ave'], info_dict['CoVariance'], info_dict['Amount'])
        return stat_dict

    def update(self, stat_dict, gain=None):
        if gain is None:    
            for i, layer_name in enumerate(stat_dict.keys()):
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current = self.coverage(self.estimator_dict)
        else:
            (delta, layer_to_update) = gain
            for layer_name in layer_to_update:
                (new_Ave, new_CoVariance, new_Amount) = stat_dict[layer_name]
                self.estimator_dict[layer_name].Ave = new_Ave
                self.estimator_dict[layer_name].CoVariance = new_CoVariance
                self.estimator_dict[layer_name].Amount = new_Amount
            self.current += delta

    def coverage(self, stat_dict):
        val = 0
        for i, layer_name in enumerate(stat_dict.keys()):
            if isinstance(stat_dict[layer_name], tuple):
                (_, CoVariance, _) = stat_dict[layer_name]
            else:
                CoVariance = stat_dict[layer_name].CoVariance
            val += self.norm(CoVariance)
        return val

    def get_cove_dict(self):
        return self.estimator_dict

    # def gain(self, stat_new):
    #     total = 0
    #     layer_to_update = []
    #     for i, layer_name in enumerate(stat_new.keys()):
    #         (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
    #         value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
    #         if value > 0:
    #             layer_to_update.append(layer_name)
    #             total += value
    #     if total > 0:
    #         return (total, layer_to_update)
    #     else:
    #         return None

    def gain(self, data):
        stat_new = self.calculate(data)
        delta = 0
        layer_to_update = []
        for i, layer_name in enumerate(stat_new.keys()):
            (new_Ave, new_CoVar, new_Amt) = stat_new[layer_name]
            value = self.norm(new_CoVar) - self.norm(self.estimator_dict[layer_name].CoVariance)
            if value > 0:
                layer_to_update.append(layer_name)
                delta += value
        if delta > 0:
            self.update(stat_new, (delta, layer_to_update))
            return True
        return False

    def norm(self, vec, mode='L1', reduction='mean'):
        m = vec.size(0)
        assert mode in ['L1', 'L2']
        assert reduction in ['mean', 'sum']
        if mode == 'L1':
            total = vec.abs().sum()
        elif mode == 'L2':
            total = vec.pow(2).sum().sqrt()
        if reduction == 'mean':
            return total / m
        elif reduction == 'sum':
            return total

    def save(self, path):
        print('Saving recorded NLC in %s...' % path)
        stat_dict = {}
        for layer_name in self.estimator_dict.keys():
            stat_dict[layer_name] = {
                'Ave': self.estimator_dict[layer_name].Ave,
                'CoVariance': self.estimator_dict[layer_name].CoVariance,
                'Amount': self.estimator_dict[layer_name].Amount
            }
        torch.save({'stat': stat_dict}, path)

    def load(self, path):
        print('Loading saved NLC from %s...' % path)
        ckpt = torch.load(path)
        stat_dict = ckpt['stat']
        for layer_name in stat_dict.keys():
            self.estimator_dict[layer_name].Ave = stat_dict[layer_name]['Ave']
            self.estimator_dict[layer_name].CoVariance = stat_dict[layer_name]['CoVariance']
            self.estimator_dict[layer_name].Amount = stat_dict[layer_name]['Amount']

class NC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.threshold = hyper
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.coverage_dict[layer_name] = torch.zeros(layer_size[0]).type(torch.BoolTensor).to(self.device)
        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            scaled_output = tool.scale(layer_output)
            mask_index = scaled_output > self.threshold
            is_covered = mask_index.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict
    
    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()

    def get_cove_dict(self):
        return self.coverage_dict

    def save(self, path):
        print('Saving recorded NC in %s...' % path)
        torch.save(self.coverage_dict, path)

    def load(self, path):
        print('Loading saved NC from %s...' % path)
        self.coverage_dict = torch.load(path)

class TKNC(Coverage):
    def init_variable(self, hyper):
        assert hyper is not None
        self.k = hyper
        self.coverage_dict = {}
        for (layer_name, layer_size) in self.layer_size_dict.items():
            num_neuron = layer_size[0]
            self.coverage_dict[layer_name] = torch.zeros(num_neuron).type(torch.BoolTensor).to(self.device)
        self.current = 0

    def calculate(self, data):
        cove_dict = {}
        layer_output_dict = tool.get_layer_output(self.model, data)
        for (layer_name, layer_output) in layer_output_dict.items():
            batch_size = layer_output.size(0)
            num_neuron = layer_output.size(1)
            # layer_output: (batch_size, num_neuron)
            _, idx = layer_output.topk(min(self.k, num_neuron), dim=1, largest=True, sorted=False)
            # idx: (batch_size, k)
            covered = torch.zeros(layer_output.size()).to(self.device)
            index = tuple([torch.LongTensor(list(range(batch_size))), idx.transpose(0, 1)])
            covered[index] = 1
            is_covered = covered.sum(0) > 0
            cove_dict[layer_name] = is_covered | self.coverage_dict[layer_name]
        return cove_dict

    def coverage(self, cove_dict):
        (cove, total) = (0, 0)
        for layer_name in cove_dict.keys():
            is_covered = cove_dict[layer_name]
            cove += is_covered.sum()
            total += len(is_covered)
        return (cove / total).item()


class LSC(SurpriseCoverage):
    def get_name(self):
        return 'LSC'

    def set_kde(self):
        for k in self.SA_cache.keys():
            if self.num_class <= 1:
                self.kde_cache[k] = gaussian_kde(self.SA_cache[k].T)
            else:
                self.kde_cache[k] = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(self.SA_cache[k])
            # The original LSC uses the `gaussian_kde` function, however, we note that this function
            # frequently crashes due to numerical issues, especially for large `num_class`.

    def calculate(self, data_tuple):
        (data_batch, label_batch) = data_tuple
        cove_set = set()
        SA_batch = []
        batch_size = label_batch.size(0)
        layer_output_dict = tool.get_layer_output(self.model, data_batch)
        for (layer_name, layer_output) in layer_output_dict.items():
            SA_batch.append(layer_output[:, self.mask_index_dict[layer_name]].view(batch_size, -1))
        SA_batch = torch.cat(SA_batch, 1).detach().cpu().numpy() # [batch_size, num_neuron]
        for i, label in enumerate(label_batch):
            SA = SA_batch[i]
            # if (np.isnan(SA).any()) or (not np.isinf(SA).any()):
            #     continue
            if self.num_class <= 1:
                lsa = np.asscalar(-self.kde_cache[int(label.cpu())].logpdf(np.expand_dims(SA, 1)))
            else:
                lsa = np.asscalar(-self.kde_cache[int(label.cpu())].score_samples(np.expand_dims(SA, 0)))
            if (not np.isnan(lsa)) and (not np.isinf(lsa)):
                cove_set.add(int(lsa / self.threshold))
        cove_set = self.coverage_set.union(cove_set)
        return cove_set


class CC(Coverage):
    '''
    Cluster-based Coverage, i.e., the coverage proposed by TensorFuzz
    '''
    def init_variable(self, hyper):
        assert hyper is not None
        self.threshold = hyper
        self.distant_dict = {}
        self.flann_dict = {}

        for (layer_name, layer_size) in self.layer_size_dict.items():
            self.flann_dict[layer_name] = FLANN()
            self.distant_dict[layer_name] = []

    def update(self, dist_dict, delta=None):
        for layer_name in self.distant_dict.keys():
            self.distant_dict[layer_name] += dist_dict[layer_name]
            self.flann_dict[layer_name].build_index(np.array(self.distant_dict[layer_name]))
        if delta:
            self.current += delta
        else:
            self.current += self.coverage(dist_dict)

    def calculate(self, data):
        layer_output_dict = tool.get_layer_output(self.model, data)
        dist_dict = {}
        for (layer_name, layer_output) in layer_output_dict.items():
            dist_dict[layer_name] = []
            for single_output in layer_output:
                single_output = single_output.cpu().numpy()
                if len(self.distant_dict[layer_name]) > 0:
                    _, approx_distances = self.flann_dict[layer_name].nn_index(np.expand_dims(single_output, 0), num_neighbors=1)
                    exact_distances = [
                        np.sum(np.square(single_output - distant_vec))
                        for distant_vec in self.distant_dict[layer_name]
                    ]
                    buffer_distances = [
                        np.sum(np.square(single_output - buffer_vec))
                        for buffer_vec in dist_dict[layer_name]
                    ]
                    nearest_distance = min(exact_distances + approx_distances.tolist() + buffer_distances)
                    if nearest_distance > self.threshold:
                        dist_dict[layer_name].append(single_output)
                else:
                    self.flann_dict[layer_name].build_index(single_output)
                    self.distant_dict[layer_name].append(single_output)
        return dist_dict

    def get_cove_dict(self):
        return self.distant_dict

    def coverage(self, dist_dict):
        total = 0
        for layer_name in dist_dict.keys():
            total += len(dist_dict[layer_name])
        return total

    # def gain(self, dist_dict):
    #     increased = self.coverage(dist_dict)
    #     return increased

    def gain(self, data):
        dist_dict = self.calculate(data)
        increased = self.coverage(dist_dict)
        if increased > 0:
            self.update(dist_dict, increased)
            return True
        return False

    def save(self, path):
        print('Saving recorded CC in %s' % path)
        torch.save(self.distant_dict, path)

    def load(self, path):
        print('Loading saved CC from %s' % path)
        self.distant_dict = torch.load(path)