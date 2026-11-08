from DiffusionGuidence.Train import train, eval
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main(model_config, data_name, eta1=0., eta2=0., is_select=True, state='eval'):
    modelConfig = {
        "state": state,  # or eval
        "epoch": 200,
        "seed": 1,
        "predict_noise": False,
        "is_select": is_select,
        "band": 1,
        "alpha": 0.1,
        "eta1": eta1,
        "eta2": eta2,
        "test_band": 64,
        "batch_size": 256,
        "T": 500,
        "channel": 128,
        "channel_mult": [1, 2, 3, 4],
        "attn": [2],
        "num_res_blocks": 2,
        "dropout": 0.15,
        "lr": 1e-4,
        "w": 2,
        "multiplier": 2.,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "img_size": 32,
        "grad_clip": 1.,
        "device": "cuda:0",  ### MAKE SURE YOU HAVE A GPU !!!
        "training_load_weight": None,
        "save_dir": "./Checkpoints_wholeImage_%s/eta1_%s_eta2_%s/" % (data_name, eta1, eta2),
        # "save_dir": "./Checkpoints_wholeImage_gulfport11/eta2%s/" % eta2,
        "test_load_weight": "ckpt_199_.pt",
        "sampled_dir": "./SampledImgs/",
        "sampledNoisyImgName": "NoisyNoGuidenceImgs.png",
        "sampledImgName": "SampledNoGuidenceImgs.png",
        "nrow": 8,
        'dataset': data_name,
        "path": "datasets/%s.mat"%data_name
    }
    if model_config is not None:
        modelConfig = model_config
    if modelConfig["state"] == "train":
        train(modelConfig)
    else:
        return eval(modelConfig)


if __name__ == '__main__':
    margin = 0
    auc_max = 0
    opt_eta1 = 0
    opt_eta2 = 0
    main(None, 'abu-airport-4', 10, 1, False, 'eval')
