import os
from typing import Dict
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from .Patch_Data import Data
from .Model import NetWork
from .Scheduler import GradualWarmupScheduler
from .Diffusion import GaussianDiffusionSampler, GaussianDiffusionSampler2, GaussianDiffusionTrainer
from .Utils import checkFile
import scipy.io as sio
from .Utils import standard
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import metrics
from .ts_generation import ts_generation
import random


def data_expand(x):
    return torch.unsqueeze(x, dim=1)


def generate_target_image(image, prior):
    alphas = torch.rand_like(image) * 0.1
    z = alphas * image + (1 - alphas) * prior
    return z


def seed_torch(seed=1):
    '''
    Keep the seed fixed thus the results can keep stable
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def cosin_similarity(x, y):
    assert x.shape[-1] == y.shape[-1]
    x_y = np.sum(np.multiply(x, y), axis=-1)
    x_norm = np.sqrt(np.sum(x ** 2, axis=-1))
    y_norm = np.sqrt(np.sum(y ** 2, axis=-1))
    sim = x_y / (x_norm * y_norm + 1e-8)
    sim = np.reshape(sim, [x.shape[0], 1, 1], order='F')
    return np.float32(sim)


def getData(path, device):
    mat = sio.loadmat(path)
    data = mat['data']
    gt = mat['map']

    # normalize and translation to tensor
    data[data < 0] = 0
    x = standard(data)
    h, w, c = x.shape
    target_spectrum = ts_generation(data, gt, 7)
    target_spectrum = np.float32(target_spectrum)

    # x1 = torch.zeros(h * w, 1, 1).to(device)
    # x2 = torch.ones(h * w, 1, 1).to(device)

    y = np.reshape(x, [-1, c], order='F')
    # z = np.tile(target_spectrum.T, [h * w, 1])

    pixel_nums = h * w
    alphas = np.random.uniform(0, 0.1, pixel_nums)
    alphas = alphas[:, None]
    z = alphas * y + (1 - alphas) * target_spectrum.T

    x1 = cosin_similarity(y, target_spectrum.T)
    x2 = cosin_similarity(z, target_spectrum.T)

    x1 = torch.from_numpy(x1).float().to(device)
    x2 = torch.from_numpy(x2).float().to(device)

    y = torch.from_numpy(y).float().to(device)
    z = torch.from_numpy(z).float().to(device)

    # combined_map = torch.cat([x1, x2], dim=0)
    # combined_cond = torch.cat([y, z], dim=0)

    return (x2, z), (x1, y)


def paintTrend(losslist, epochs=100, stride=10):
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.title('loss-trend')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xticks(np.arange(0, epochs, stride))
    plt.xlim(0, epochs)
    plt.plot(losslist, color='r')
    plt.show()


def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    seed_torch(modelConfig['seed'])
    # dataset
    dataset = Data(modelConfig["path"])
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)
    # model setup
    net_model = NetWork(T=modelConfig["T"], band=1, test_band=dataset.band,
                        ch=modelConfig["channel"], dropout=modelConfig["dropout"])
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["training_load_weight"]), map_location=device), strict=False)
        print("Model weight load down.")
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(optimizer=optimizer, multiplier=modelConfig["multiplier"],
                                             warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        predict_noise=modelConfig['predict_noise'], batch_size=modelConfig['batch_size']).to(device)
    checkFile(modelConfig["save_dir"])
    # start training
    loss_list = []
    net_model.train()
    # //
    # iter_batch_sizes = np.linspace(modelConfig['batch_size'], 10000, modelConfig["epoch"])
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for positives, negatives in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                positive_data = positives[0].to(device)
                positive_label = positives[1].to(device)
                negative_data = negatives[0].to(device)
                negative_label = negatives[1].to(device)

                data = torch.cat([positive_data, negative_data], dim=0)
                label = torch.cat([positive_label, negative_label], dim=0)

                data = data * 2 - 1
                label = label * 2 - 1

                loss = trainer(data, label, eta1=modelConfig['eta1'], eta2=modelConfig['eta2'])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    # "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                    "BS": dataloader.batch_size
                })
        warmUpScheduler.step()
        loss_list.append(loss.item())
    torch.save(net_model.state_dict(), os.path.join(
        modelConfig["save_dir"], 'ckpt_' + str(e) + "_.pt"))
    paintTrend(loss_list, epochs=modelConfig['epoch'], stride=20)


def pca_manipulate(x, n_components=30):
    pca = PCA(n_components)
    pca_x = pca.fit_transform(x)
    return pca_x


def eval(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    seed_torch(modelConfig['seed'])
    # load model and evaluate
    with torch.no_grad():
        mat = sio.loadmat(modelConfig['path'])
        data = mat['data']
        gt = mat['map']
        data = standard(data)
        # data = data[30:40, 50:60, :]
        # gt = gt[30:40, 50:60]
        h, w, band = data.shape
        # gt = mat['map']
        data = np.reshape(data, [-1, band], order='F')
        data = 2 * data - 1
        data = torch.from_numpy(data).float()
        data = data.to(device)
        # print(data.shape)
        # model = spectral_UNet(T=modelConfig["T"], band=modelConfig['band'], test_band=modelConfig['test_band'],
        #                       ch=modelConfig["channel"],
        #                       ch_mult=modelConfig["channel_mult"],
        #                       num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
        # model = Network({'in_channels': modelConfig['band'], 'pre_embed_dim': modelConfig['channel'], 'hidden_dim': [200, 100, 50]}).to(device)
        model = NetWork(T=modelConfig["T"], band=1, test_band=band,
                        ch=modelConfig["channel"],
                        dropout=modelConfig["dropout"])
        ckpt = torch.load(os.path.join(
            modelConfig["save_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        if modelConfig['predict_noise']:
            sampler = GaussianDiffusionSampler(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        else:
            sampler = GaussianDiffusionSampler2(
                model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"], w=modelConfig["w"]).to(device)
        numpixel = h * w
        noisySpectra = torch.randn(
            size=[numpixel, 1], device=device)
        print(noisySpectra.shape)
        sampledSpectra = sampler(noisySpectra, data)
        sampledSpectra = sampledSpectra * 0.5 + 0.5  # [0 ~ 1] ##还原为[0,1]
        sampledSpectra = torch.squeeze(sampledSpectra)
        sampledSpectra = sampledSpectra.cpu().detach().numpy()
        # detection_map = np.sqrt(np.sum(sampledSpectra ** 2, axis=-1))
        detection_map = np.reshape(sampledSpectra, [h, w], order='F')
        detection_map = standard(detection_map)
        # lamba = 0.05
        # detection_map = np.exp(-1 * (detection_map - 1) **2 / lamba)
        plt.imshow(detection_map)
        plt.show()
        y_l = np.reshape(gt, [-1, 1], order='F')
        y_p = np.reshape(detection_map, [-1, 1], order='F')

        ## calculate the AUC value
        fpr, tpr, threshold = metrics.roc_curve(y_l, y_p, drop_intermediate=False)
        fpr = fpr[1:]
        tpr = tpr[1:]
        threshold = threshold[1:]
        epision = 6
        try:
            auc1 = round(metrics.auc(fpr, tpr), epision)
            # auc2 = round(metrics.auc(threshold, fpr), epision)
            # auc3 = round(metrics.auc(threshold, tpr), epision)
            # auc4 = round(auc1 + auc3 - auc2, epision)
            # auc5 = round(auc3 / auc2, epision)
        except:
            auc1 = 0.5
        if modelConfig['is_select']:
            return auc1
        auc2 = round(metrics.auc(threshold, fpr), epision)
        auc3 = round(metrics.auc(threshold, tpr), epision)
        auc4 = round(auc1 + auc3 - auc2, epision)
        auc5 = round(auc3 / auc2, epision)
        print('{:.{precision}f}'.format(auc1, precision=epision))
        print('{:.{precision}f}'.format(auc2, precision=epision))
        print('{:.{precision}f}'.format(auc3, precision=epision))
        print('{:.{precision}f}'.format(auc4, precision=epision))
        print('{:.{precision}f}'.format(auc5, precision=epision))

        # plt.imshow(gt)
        # plt.show()
        # plt.imsave('%s_map.png'%modelConfig['dataset'],detection_map,cmap='gray',dpi=600)
        # plt.imsave('%s_gt.png' % modelConfig['dataset'], gt, cmap='gray', dpi=600)
