import argparse
import os
import json
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Function
import torch

class TripletMarginLoss:
    """Triplet loss function.
    """
    def __init__(self, margin):
        self.margin = margin
        self.pdist = PairwiseDistance(2)  # norm 2

    def apply(self, anchor, positive, negative):
        d_p = self.pdist.apply(anchor, positive)
        d_n = self.pdist.apply(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0)
        loss = torch.mean(dist_hinge)
        return loss

class PairwiseDistance:
    def __init__(self, p):
        self.norm = p

    def apply(self, x1, x2):
        assert x1.size() == x2.size()
        eps = 1e-4 / x1.size(1)
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, self.norm).sum(dim=1)
        return torch.pow(out + eps, 1. / self.norm)

class CelebA(Dataset):
    def __init__(self, 
                img_dir='/export/d3/user/dataset/celeba_crop128/',
                name2id_path='/export/d3/user/dataset/celeba_crop128/CelebA_name_to_ID.json',
                id2name_path='/export/d3/user/dataset/celeba_crop128/CelebA_ID_to_name.json',
                split='train',
                img_size=64):
        assert split in ['train', 'val', 'test']
        self.transform = transforms.Compose([
                       transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()
               ])
        with open(name2id_path, 'r') as f:
            self.name2id = json.load(f)
        with open(id2name_path, 'r') as f:
            self.id2name = json.load(f)
        self.name_dir = img_dir + split + '/'
        self.name_list = sorted(os.listdir(self.name_dir))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        im = Image.open(self.name_dir + self.name_list[index])
        tsr = self.transform(im)
        c = torch.LongTensor([0])
        return tsr, c.squeeze()

class TripletCelebA(Dataset):
    def __init__(self, 
                img_dir='/export/d3/user/dataset/celeba_crop128/',
                name2id_path='/export/d3/user/dataset/celeba_crop128/CelebA_name_to_ID.json',
                id2name_path='/export/d3/user/dataset/celeba_crop128/CelebA_ID_to_name.json',
                split='train',
                img_size=64,
                n_triplets=100):
        self.n_triplets = n_triplets
        assert split in ['train', 'val', 'test']
        self.transform = transforms.Compose([
                       transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()
               ])
        with open(name2id_path, 'r') as f:
            self.name2id = json.load(f)
        with open(id2name_path, 'r') as f:
            self.id2name = json.load(f)
        self.name_dir = img_dir + split + '/'
        self.name_list = sorted(os.listdir(self.name_dir))
        self.ID_list = [int(self.name2id[name]) for name in self.name_list]
        self.n_class = np.max(self.ID_list) + 1

        self.triplets_list = self.generate_triplets(self.ID_list, self.n_triplets)

    def generate_triplets(self, y_label, num_triplets):
        def create_indices(y_label):
            inds = dict()
            for idx, label in enumerate(y_label):
                if label not in inds.keys():
                    inds[label] = []
                inds[label].append(idx)
            return inds

        triplets = []
        # Indices = array of labels and each label is an array of indices
        indices = create_indices(y_label)

        label_set = set(y_label)
        print('num_classes:', len(label_set))
        for i in range(num_triplets):
            c1 = random.sample(label_set, 1)[0]
            c2 = random.sample(label_set, 1)[0]
            while len(indices[c1]) < 2:
                c1 = random.sample(label_set, 1)[0]

            while c1 == c2:
                c2 = random.sample(label_set, 1)[0]
            if len(indices[c1]) == 2:  # hack to speed up process
                n1, n2 = 0, 1
            else:
                n1 = np.random.randint(0, len(indices[c1]) - 1)
                n2 = np.random.randint(0, len(indices[c1]) - 1)
                while n1 == n2:
                    n2 = np.random.randint(0, len(indices[c1]) - 1)
            if len(indices[c2]) ==1:
                n3 = 0
            else:
                n3 = np.random.randint(0, len(indices[c2]) - 1)

            triplets.append([indices[c1][n1], indices[c1][n2], indices[c2][n3], c1, c2])
            if i and i % 50000 == 0:
                print('Created %d triplets...' % i)
        return triplets

    def __getitem__(self, index):
        a, p, n, c1, c2 = self.triplets_list[index]
        im_a = Image.open(self.name_dir + self.name_list[a])
        im_p = Image.open(self.name_dir + self.name_list[p])
        im_n = Image.open(self.name_dir + self.name_list[n])

        tsr_a = self.transform(im_a)
        tsr_p = self.transform(im_p)
        tsr_n = self.transform(im_n)
        c1 = torch.LongTensor([c1])
        c2 = torch.LongTensor([c2])
        return tsr_a, tsr_p, tsr_n, c1.squeeze(), c2.squeeze()

    def __len__(self):
        return len(self.triplets_list)


class FaceNet(nn.Module):
    def __init__(self, n_class, nc=3, dim=100, alpha=1):
        super(FaceNet, self).__init__()
        nf = 32
        self.dim = dim
        self.alpha = alpha
        self.net = nn.Sequential(
            # 64 x 64
            nn.Conv2d(nc, nf, 4, 2, 1),
            nn.BatchNorm2d(nf),
            nn.ReLU(),
            # 32 x 32
            nn.Conv2d(nf, nf * 2, 4, 2, 1),
            nn.BatchNorm2d(nf * 2),
            nn.ReLU(),
            # 16 x 16
            nn.Conv2d(nf * 2, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(),
            # 8 x 8
            nn.Conv2d(nf * 4, nf * 4, 4, 2, 1),
            nn.BatchNorm2d(nf * 4),
            nn.ReLU(),
            # 4 x 4
            nn.Conv2d(nf * 4, dim, 4, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            # 1 x 1
        )
        self.fc = nn.Sequential(
                nn.Linear(dim, n_class),
                nn.Sigmoid()
            )

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        h = self.net(x)
        bs = h.size(0)
        h = h.view(bs, -1)
        feat = self.l2_norm(h)
        out = self.fc(feat)
        return feat * self.alpha, out


# os.makedirs("images", exist_ok=True)
# os.makedirs("ckpt", exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='face_recog_32', help="experiment name")
    parser.add_argument("--cls", type=int, default=0, help="selected class")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--num_tuple", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=3, help="number of training steps for discriminator per iter")
    parser.add_argument("--lambda_gp", type=float, default=10, help="loss weight for gradient penalty")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    # parser.add_argument("--sample_interval", type=int, default=500, help="interval betwen image samples")
    parser.add_argument("--save_every", type=int, default=5, help="interval betwen image samples")
    opt = parser.parse_args()
    print(opt)

    # os.makedirs("images/%s" % opt.exp_name, exist_ok=True)
    # os.makedirs("ckpt/%s" % opt.exp_name, exist_ok=True)

    img_shape = (opt.channels, opt.img_size, opt.img_size)

    cuda = True if torch.cuda.is_available() else False

    # Configure data loader

    train_set = TripletCelebA(split='train', n_triplets=opt.num_tuple, img_size=opt.img_size)
    test_set = TripletCelebA(split='test', n_triplets=opt.num_tuple, img_size=opt.img_size)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    classifier = FaceNet(n_class=max(train_set.n_class, test_set.n_class))
    # classifier = RecogSeq32()

    if cuda:
        classifier.cuda()

    # Optimizers
    optimizer = torch.optim.Adam(classifier.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


    # ----------
    #  Training
    # ----------

    # def accuracy(pred, target):
    #     is_same = ((pred > 0.5) == target)
    #     return (is_same.sum() / len(is_same)).item()

    def accuracy(preds, y):
        preds = torch.argmax(preds, dim=1)
        correct = (preds == y).float()
        if len(correct) == 0:
            return 0
        #print(correct)
        #print(correct.shape)
        acc = correct.sum().item() / len(correct)
        return acc

    mse = nn.MSELoss().cuda()
    bce_log = nn.BCEWithLogitsLoss().cuda()
    bce = nn.BCELoss().cuda()
    ce = nn.CrossEntropyLoss().cuda()

    l2_dist = PairwiseDistance(2)

    def train(margin=0.5):
        tml = TripletMarginLoss(margin)
        loss_list = []
        acc_cls_list = []
        acc_triplet_list = []
        classifier.train()
        for i, (data_a, data_p, data_n, label_p, label_n) in enumerate(tqdm(train_loader)):
            # Configure input
            data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
            label_p, label_n = label_p.cuda(), label_n.cuda()

            out_a, cls_a = classifier(data_a)
            out_p, cls_p = classifier(data_p)
            out_n, cls_n = classifier(data_n)

            # choose hard neg
            d_p = l2_dist.apply(out_a, out_p)
            d_n = l2_dist.apply(out_a, out_n)

            less = (d_p - d_n < 0)
            triplets_index = (d_n - d_p < margin).nonzero(as_tuple=True)[0]

            triplet_loss = tml.apply(out_a[triplets_index], out_p[triplets_index], out_n[triplets_index])
            predicted_labels = torch.cat([cls_a[triplets_index], cls_p[triplets_index], cls_n[triplets_index]])
            true_labels = torch.cat([label_p[triplets_index], label_p[triplets_index], label_n[triplets_index]])

            cross_entropy_loss = ce(predicted_labels, true_labels)

            loss = cross_entropy_loss + triplet_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_cls = accuracy(predicted_labels, true_labels)
            acc_triplet = less.sum().item() / len(less)

            loss_list.append(loss.item())
            acc_cls_list.append(acc_cls)
            acc_triplet_list.append(acc_triplet)
        return loss_list, acc_cls_list, acc_triplet_list

    def test(margin=0.5):
        tml = TripletMarginLoss(margin)
        with torch.no_grad():
            acc_cls_list = []
            acc_triplet_list = []
            classifier.eval()
            for i, (data_a, data_p, data_n, label_p, label_n) in enumerate(tqdm(train_loader)):
                # Configure input
                data_a, data_p, data_n = data_a.cuda(), data_p.cuda(), data_n.cuda()
                label_p, label_n = label_p.cuda(), label_n.cuda()

                out_a, cls_a = classifier(data_a)
                out_p, cls_p = classifier(data_p)
                out_n, cls_n = classifier(data_n)

                # choose hard neg
                d_p = l2_dist.apply(out_a, out_p)
                d_n = l2_dist.apply(out_a, out_n)

                less = (d_p - d_n < 0)
                triplets_index = (d_n - d_p < margin).nonzero(as_tuple=True)[0]
                
                triplet_loss = tml.apply(out_a[triplets_index], out_p[triplets_index], out_n[triplets_index])
                predicted_labels = torch.cat([cls_a[triplets_index], cls_p[triplets_index], cls_n[triplets_index]])
                
                true_labels = torch.cat([label_p[triplets_index], label_p[triplets_index], label_n[triplets_index]])
                cross_entropy_loss = ce(predicted_labels, true_labels)

                loss = cross_entropy_loss + triplet_loss
                acc_cls = accuracy(predicted_labels, true_labels)
                acc_triplet = less.sum().item() / len(less)

                loss_list.append(loss.item())
                acc_cls_list.append(acc_cls)
                acc_triplet_list.append(acc_triplet)
            return loss_list, acc_cls_list, acc_triplet_list

    for epoch in range(opt.n_epochs):
        loss_list, acc_cls_list, acc_triplet_list = train()
        print(
            "[Epoch %d/%d] [loss: %f] [pred acc: %f] [triplets acc: %f]"
            % (epoch, opt.n_epochs, np.mean(loss_list), np.mean(acc_cls_list), np.mean(acc_triplet_list))
        )
        if epoch % opt.save_every == 0:
            loss_list, acc_cls_list, acc_triplet_list = test()
            print(
                "[Test] [loss: %f] [pos acc: %f] [neg acc: %f]"
                % (np.mean(loss_list), np.mean(acc_cls_list), np.mean(acc_triplet_list))
            )
            # state = {
            #     'classifier': classifier.state_dict(),
            #     'optimizer': optimizer.state_dict(),
            # }
            # torch.save(state, "ckpt/%s/%d.ckpt" % (opt.exp_name, epoch))

    loss_list, acc_cls_list, acc_triplet_list = test()
    print(
        "[Test] [loss: %f] [pos acc: %f] [neg acc: %f]"
        % (np.mean(loss_list), np.mean(acc_cls_list), np.mean(acc_triplet_list))
    )
    state = {
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, "trained_models/face.pt")