import torch
import torch.nn as nn
import torch.nn.functional as F
from .Utils import standard
import numpy as np
import matplotlib.pyplot as plt

def extract(v, t, x_shape):
    """
    Extract some coefficients at specified timesteps, then reshape to
    [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
    """
    device = t.device
    out = torch.gather(v, index=t, dim=0).float().to(device)
    return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class GaussianDiffusionTrainer(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, predict_noise, batch_size):
        super().__init__()

        self.model = model
        self.T = T
        self.predict_noise = predict_noise
        self.batch_size = batch_size

        self.register_buffer(
            'betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', torch.sqrt(alphas_bar))
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))
        self.register_buffer(
            'alphas_bar_prev', alphas_bar_prev)
        self.register_buffer(
            'alphas_bar', alphas_bar)
        self.register_buffer(
            'alphas', alphas)

    def forward(self, x_0, labels, eta1=0., eta2=0.):
        """
        Algorithm 1.
        """
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 + \
              extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
        lambd = 0.5 * extract(self.alphas_bar_prev, t, x_0.shape) * (1-extract(self.alphas, t, x_0.shape)) / (1-extract(self.alphas_bar, t, x_0.shape))**2
        # print(lambd.shape)
        # exit(-1)
        # x_t = torch.zeros_like(x_t).cuda()
        # t = torch.zeros_like(t).cuda()
        # full_t = torch.randint(self.T, size=(full_map.shape[0],), device=full_map.device)
        # noise_full_t = torch.randn_like(full_map)
        # full_x_t = extract(self.sqrt_alphas_bar, full_t, full_map.shape) * full_map + \
        #            extract(self.sqrt_one_minus_alphas_bar, full_t, full_map.shape) * noise_full_t
        if self.predict_noise:
            loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='none')
        else:
            coarse_x_0 = self.model(x_t, t, labels)
            coarse_x_02 = (coarse_x_0 + 1) / 2
            batch_size = x_0.shape[0] // 2
            # triplet_loss = 1.0 + coarse_x_0[batch_size:] - coarse_x_0[:batch_size]
            # triplet_loss = torch.relu(triplet_loss)
            # triplet_loss = torch.mean(triplet_loss)
            # p_sim = torch.exp(coarse_x_0[:self.batch_size] / tau)
            # n_sim = torch.exp(coarse_x_0[self.batch_size:] / tau)
            # info_loss = torch.mean(-1 * torch.log(p_sim / (p_sim + n_sim)))
            # constrained_loss = coarse_x_0[self.batch_size:] ** 2
            # constrained_loss = torch.mean(constrained_loss)
            p_sim = torch.clip(coarse_x_02[:self.batch_size], 0, 1)
            n_sim = torch.clip(1 - coarse_x_02[self.batch_size:], 0, 1)
            # p_sim =  torch.sigmoid(coarse_x_02[:self.batch_size])
            # n_sim =  torch.sigmoid(1 - coarse_x_02[self.batch_size:])
            bce_loss = -0.5 * torch.mean(torch.log(p_sim + 1e-8) + torch.log(n_sim + 1e-8))
            cem_loss = coarse_x_02[batch_size:] ** 2 + eta1 * torch.abs(coarse_x_02[:batch_size] - 1)
            cem_loss = torch.mean(cem_loss)
            # p_loss = torch.binary_cross_entropy_with_logits(coarse_x_02[:self.batch_size], torch.ones_like(coarse_x_02[:self.batch_size]))
            # n_loss = torch.binary_cross_entropy_with_logits(coarse_x_02[self.batch_size:], torch.ones_like(coarse_x_02[self.batch_size:]))
            # bce_loss = 0.5 * (torch.mean(p_loss) + torch.mean(n_loss))
            # sparse_loss = torch.abs(coarse_x_0[self.batch_size:])
            # sparse_loss = torch.mean(sparse_loss)
            # recon_loss = lambd[:batch_size]*torch.square(coarse_x_0[:batch_size]-x_0[:batch_size]) + lambd[batch_size:]*torch.square(coarse_x_0[batch_size:]-x_0[batch_size:])
            # print(recon_loss.shape)
            # recon_loss = torch.square(coarse_x_0-x_0)
            # recon_loss = torch.mean(recon_loss)
            # loss = cem_loss + eta2 * recon_loss
            # loss = cem_loss + eta2 * recon_loss
            # diffusion_loss = F.mse_loss(self.model(x_t, t, labels), noise, reduction='mean')
            loss = cem_loss + eta2 * bce_loss
        # coarse_x_0 = self.model(x_t, t, labels)
        # # eta = 1
        # # loss = coarse_x_0[1] ** 2 + eta * torch.abs(coarse_x_0[0] - 1)
        # margin = 0.8
        # loss = torch.relu(margin + coarse_x_0[1] - coarse_x_0[0])
        return loss


class GaussianDiffusionSampler(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        eps = self.model(x_t, t, labels)
        xt_prev_mean = self.predict_xt_prev_mean_from_eps(x_t, t, eps=eps)
        return xt_prev_mean, var

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        return torch.clip(x_0, -1, 1)


class GaussianDiffusionSampler2(nn.Module):
    def __init__(self, model, beta_1, beta_T, T, w=0.):
        super().__init__()

        self.model = model
        self.T = T
        ### In the classifier free guidence paper, w is the key to control the gudience.
        ### w = 0 and with label = 0 means no guidence.
        ### w > 0 and label > 0 means guidence. Guidence would be stronger if w is bigger.
        self.w = w

        self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).double())
        alphas = 1. - self.betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]
        self.register_buffer('alphas_bar', alphas_bar)
        self.register_buffer('coeff1', torch.sqrt(1. / alphas))
        self.register_buffer('coeff2', self.coeff1 * (1. - alphas) / torch.sqrt(1. - alphas_bar))
        self.register_buffer('posterior_var', self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))

        self.register_buffer('coeff3', torch.sqrt(alphas_bar_prev) * (1. - alphas) / (1. - alphas_bar))
        self.register_buffer('coeff4', torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

    def predict_xt_prev_mean_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return extract(self.coeff1, t, x_t.shape) * x_t - extract(self.coeff2, t, x_t.shape) * eps

    def predict_xt_prev_mean_from_x0(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return extract(self.coeff3, t, x_0.shape) * x_0 + extract(self.coeff4, t, x_t.shape) * x_t

    def ddim_p_mean_variance(self, x_t, t, pre_t, labels, eta=0):
        pred_x0 = self.model(x_t, t, labels)
        pred_x0 = torch.clamp(pred_x0, -1, 1)
        alpha_bar = extract(self.alphas_bar, t, x_t.shape)
        pred_eps = (x_t - torch.sqrt(alpha_bar) * pred_x0) / torch.sqrt(1 - alpha_bar)
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = eta * extract(var, t, x_t.shape)
        sigma = torch.sqrt(var)
        pred_alpha_bar = extract(self.alphas_bar, pre_t, x_t.shape)
        x_t = torch.sqrt(pred_alpha_bar) * pred_x0 + torch.sqrt(1 - pred_alpha_bar - var) * pred_eps + sigma * torch.randn_like(x_t)
        return x_t, pred_x0

    def p_mean_variance(self, x_t, t, labels):
        # below: only log_variance is used in the KL computations
        var = torch.cat([self.posterior_var[1:2], self.betas[1:]])
        var = extract(var, t, x_t.shape)
        x_0 = self.model(x_t, t, labels)
        # nonEps = self.model(x_t, t, torch.zeros_like(labels).to(labels.device))
        # x_0 = nonEps + self.w * (x_0 - nonEps)
        xt_prev_mean = self.predict_xt_prev_mean_from_x0(x_t, t, x_0)
        return xt_prev_mean, var

    def forward3(self, x_T, labels):
        x_t = torch.zeros_like(x_T).cuda()
        t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * 0
        x_0 = self.model(x_t, t, labels)
        return torch.clip(x_0, -1, 1)

    def forward(self, x_T, labels):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            print(time_step)
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step
            mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            if time_step > 0:
                noise = torch.randn_like(x_t)
            else:
                noise = 0
            x_t = mean + torch.sqrt(var) * noise
            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * (self.T - 1)
        # x_0 = self.model(x_T, t, labels)
        return torch.clip(x_0, -1, 1)

    def forward(self, x_T, labels, ddim_time_step=5, discr_method='quad'):
        """
        Algorithm 2.
        """
        x_t = x_T
        ## making sampled timestep sequence
        if discr_method == 'linear':
            c = self.T // ddim_time_step
            ddim_timestep_seq = np.asarray(list(range(0, self.T, c)))
        elif discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(self.T * .8), ddim_time_step)) ** 2
            ).astype(int)
        else:
            raise NotImplementedError(f'There is no ddim discretization method called "{discr_method}"')
        # add one to get the final alpha values right (the ones from first scale to data during sampling)
        ddim_timestep_seq = ddim_timestep_seq + 1
        # previous sequence
        batch_size = x_T.shape[0]
        device = torch.device("cuda:0")
        ddim_timestep_prev_seq = np.append(np.array([0]), ddim_timestep_seq[:-1])
        for i in reversed(range(0, ddim_time_step)):
            print(ddim_timestep_seq[i])
            t = torch.full((batch_size,), ddim_timestep_seq[i], device=device, dtype=torch.long)
            pre_t = torch.full((batch_size,), ddim_timestep_prev_seq[i], device=device, dtype=torch.long)
            # mean, var = self.p_mean_variance(x_t=x_t, t=t, labels=labels)
            # if time_step > 0:
            #     noise = torch.randn_like(x_t)
            # else:
            #     noise = 0
            # x_t = mean + torch.sqrt(var) * noise
            x_t, pre_x0 = self.ddim_p_mean_variance(x_t, t, pre_t, labels, eta=0)
            x_t = torch.clamp(x_t, -1, 1)

            # sampledSpectra = pre_x0
            # sampledSpectra = sampledSpectra * 0.5 + 0.5  # [0 ~ 1] ##还原为[0,1]
            # sampledSpectra = torch.squeeze(sampledSpectra)
            # sampledSpectra = sampledSpectra.cpu().detach().numpy()
            # # detection_map = np.sqrt(np.sum(sampledSpectra ** 2, axis=-1))
            # detection_map = np.reshape(sampledSpectra, [100, 100], order='F')
            # detection_map = standard(detection_map)
            # plt.imsave('%s_x0.png' % i, detection_map, dpi=600)

            assert torch.isnan(x_t).int().sum() == 0, "nan in tensor."
        x_0 = x_t
        # t = x_T.new_ones([x_T.shape[0], ], dtype=torch.long) * (self.T - 1)
        # x_0 = self.model(x_T, t, labels)
        return torch.clip(x_0, -1, 1)