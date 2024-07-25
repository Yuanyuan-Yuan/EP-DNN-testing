from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.functional import jacobian


class RobustPCA(object):
    """This script performing low rank factorization on the input
    matrix by Principal Component Pursuit using ADMM.

    M = L + S
        s.t. L is of low-rank,
             S is sparse.
    where M = J.t * J in our paper.

    M = ||L||_* + lambda * ||S||_1
        s.t. M = L + S

    Setting for algorithms:

    (1) M is the J.t * J, where J is jacobian matrix
        obtained from a specific region.
    (2) lamb is the coefficient to balance the low-rank and sparsity

    The details of this algorithm can refer to this link
    https://book-wright-ma.github.io/
    """

    def __init__(self, M, lamb=1/60):
        """Initializes with the matrix to perform low-rank factorization on."""
        self.M = M
        self.S = np.zeros(self.M.shape)     # sparse matrix
        self.L = np.zeros(self.M.shape)     # low-rank matrix
        self.Lamb = np.zeros(self.M.shape)  # Lambda matrix
        # mu is the coefficient used in augmented Lagrangian.
        self.mu = np.prod(self.M.shape) / (4 * np.linalg.norm(self.M, ord=1))
        self.mu_inv = 1 / self.mu
        self.iter = 0
        self.error = 1e-7 * self.frobenius_norm(self.M)

        if lamb:
            self.lamb = lamb
        else:
            self.lamb = 1 / np.sqrt(np.max(self.M.shape))

    def reset_iter(self):
        """Resets the iteration."""
        self.iter = 0

    @staticmethod
    def frobenius_norm(M):
        """Computes the Frobenius norm of a given matrix."""
        return np.linalg.norm(M, ord='fro')

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, VH = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), VH))

    def fit(self, max_iter=10000, iter_print=100):
        self.reset_iter()
        err_i = np.Inf
        S_k = self.S
        L_k = self.L
        Lamb_k = self.Lamb

        while (err_i > self.error) and self.iter < max_iter:
            L_k = self.svd_threshold(
                self.M - S_k - self.mu_inv * Lamb_k, self.mu_inv)
            S_k = self.shrink(
                self.M - L_k - self.mu_inv * Lamb_k, self.mu_inv * self.lamb)
            Lamb_k = Lamb_k + self.mu * (L_k + S_k - self.M)
            err_i = self.frobenius_norm(L_k + S_k - self.M)
            self.iter += 1
            # if (self.iter % iter_print) == 0:
            #     print(f'iteration: {self.iter}, error: {err_i}')

        return L_k, S_k

class RobustPCA_Torch(object):
    def __init__(self, M, lamb=1/60):
        """Initializes with the matrix to perform low-rank factorization on."""
        self.M = M
        self.S = torch.zeros(self.M.shape).cuda()     # sparse matrix
        self.L = torch.zeros(self.M.shape).cuda()     # low-rank matrix
        self.Lamb = torch.zeros(self.M.shape).cuda()  # Lambda matrix
        # mu is the coefficient used in augmented Lagrangian.
        self.mu = np.prod(list(self.M.shape)) / (4 * torch.norm(self.M, p=1))
        self.mu_inv = 1 / self.mu
        self.iter = 0
        self.error = 1e-7 * self.frobenius_norm(self.M)

        if lamb:
            self.lamb = lamb
        else:
            self.lamb = 1 / torch.sqrt(torch.max(self.M.shape))

    def reset_iter(self):
        """Resets the iteration."""
        self.iter = 0

    @staticmethod
    def frobenius_norm(M):
        """Computes the Frobenius norm of a given matrix."""
        return torch.norm(M, p='fro')

    @staticmethod
    def shrink(M, tau):
        return torch.sign(M) * torch.maximum((torch.abs(M) - tau), torch.zeros(M.shape).cuda())

    def svd_threshold(self, M, tau):
        U, S, VH = torch.linalg.svd(M, full_matrices=False)
        return torch.dot(U, torch.dot(torch.diag(self.shrink(S, tau)), VH))

    def fit(self, max_iter=10000, iter_print=100):
        self.reset_iter()
        err_i = np.Inf
        S_k = self.S
        L_k = self.L
        Lamb_k = self.Lamb

        while (err_i > self.error) and self.iter < max_iter:
            L_k = self.svd_threshold(
                self.M - S_k - self.mu_inv * Lamb_k, self.mu_inv)
            S_k = self.shrink(
                self.M - L_k - self.mu_inv * Lamb_k, self.mu_inv * self.lamb)
            Lamb_k = Lamb_k + self.mu * (L_k + S_k - self.M)
            err_i = self.frobenius_norm(L_k + S_k - self.M)
            self.iter += 1
            # if (self.iter % iter_print) == 0:
            #     print(f'iteration: {self.iter}, error: {err_i}')

        return L_k, S_k


def direction(jacobians, save_dir=None,
        foreground_ind=None,
        background_ind=None,
        lamb=60,
        num_relax=0,
        max_iter=10000):
    # lamb: the coefficient to control the sparsity
    # num_relax: factor of relaxation for the non-zeros singular values
    image_size = jacobians.shape[2]
    z_dim = jacobians.shape[-1]
    for ind in tqdm(range(jacobians.shape[0])):
        jacobian = jacobians[ind]
        if foreground_ind is not None and background_ind is not None:
            if len(jacobian.shape) == 4:  # [H, W, 1, latent_dim]
                jaco_fore = jacobian[foreground_ind[0], foreground_ind[1], 0]
                jaco_back = jacobian[background_ind[0], background_ind[1], 0]
            elif len(jacobian.shape) == 5:  # [channel, H, W, 1, latent_dim]
                jaco_fore = jacobian[:, foreground_ind[0], foreground_ind[1], 0]
                jaco_back = jacobian[:, background_ind[0], background_ind[1], 0]
            else:
                raise ValueError(f'Shape of Jacobian is not correct!')
            jaco_fore = np.reshape(jaco_fore, [-1, z_dim])
            jaco_back = np.reshape(jaco_back, [-1, z_dim])
            coef_f = 1 / jaco_fore.shape[0]
            coef_b = 1 / jaco_back.shape[0]
            M_fore = coef_f * jaco_fore.T.dot(jaco_fore)
            B_back = coef_b * jaco_back.T.dot(jaco_back)
            # low-rank factorization on foreground
            RPCA = RobustPCA(M_fore, lamb=1/lamb)
            L_f, _ = RPCA.fit(max_iter=max_iter)
            rank_f = np.linalg.matrix_rank(L_f)
            print('rank_f: ', rank_f)
            # low-rank factorization on background
            RPCA = RobustPCA(B_back, lamb=1/lamb)
            L_b, _ = RPCA.fit(max_iter=max_iter)
            rank_b = np.linalg.matrix_rank(L_b)
            # SVD on the low-rank matrix
            _, _, VHf = np.linalg.svd(L_f)
            _, _, VHb = np.linalg.svd(L_b)
            F_principal = VHf[:rank_f]  # Principal space of foreground
            relax_subspace = min(max(1, rank_b - num_relax), z_dim-1)
            B_null = VHb[relax_subspace:].T  # Null space of background

            F_principal_proj = B_null.dot(B_null.T).dot(F_principal.T)  # Projection
            F_principal_proj = F_principal_proj.T
            F_principal_proj /= np.linalg.norm(
                F_principal_proj, axis=1, keepdims=True)
            print('direction size: ', F_principal_proj.shape)
            save_name = '%d_direction.npy' % ind
            np.save(save_dir + save_name, F_principal_proj)
            return F_principal_proj
        else:
            jaco = np.reshape(jacobian, [-1, z_dim])
            coef = 1 / jaco.shape[0]
            M = coef * jaco.T.dot(jaco)

            RPCA = RobustPCA(M, lamb=1/lamb)
            L, _ = RPCA.fit(max_iter=max_iter)
            rank = np.linalg.matrix_rank(L)
            print('rank: ', rank)
            _, _, VH = np.linalg.svd(L)
            principal = VH[:max(rank, 5)]
            print('direction size: ', principal.shape)
            if save_dir is not None:    
                save_name = '%d_direction.npy' % ind
                np.save(save_dir + save_name, principal)
            return principal

def Jacobian(G, latent_zs):
    jacobians = []
    for idx in range(latent_zs.shape[0]):
        latent_z = latent_zs[idx:idx+1]
        jac_i = jacobian(
                    func=G,
                    inputs=latent_z,
                    create_graph=False,
                    strict=False
                )
        # print('jac_i: ', jac_i.size())
        jacobians.append(jac_i)
    jacobians = torch.cat(jacobians, dim=0)
    # print('jacobians size: ', jacobians.size())
    np_jacobians = jacobians.detach().cpu().numpy()
    return np_jacobians

def Jacobian_Y(G, latent_zs, ys):
    jacobians = []
    for idx in range(latent_zs.shape[0]):
        latent_z = latent_zs[idx:idx+1]
        y = ys[idx:idx+1]
        jac_i = jacobian(
                    func=G,
                    inputs=(latent_z, y),
                    create_graph=False,
                    strict=False
                )
        # print('jac_i: ', jac_i.size())
        jacobians.append(jac_i[0])
    jacobians = torch.cat(jacobians, dim=0)
    np_jacobians = jacobians.detach().cpu().numpy()
    return np_jacobians
