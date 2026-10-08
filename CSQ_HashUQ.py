from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard  # direct import  hadamrd matrix from scipy
import argparse
import random
import pandas as pd
from tqdm import tqdm
from functools import partialmethod
import math

torch.multiprocessing.set_sharing_strategy('file_system')


# CSQ(CVPR2020)
# paper [Central Similarity Quantization for Efficient Image and Video Retrieval](https://openaccess.thecvf.com/content_CVPR_2020/papers/Yuan_Central_Similarity_Quantization_for_Efficient_Image_and_Video_Retrieval_CVPR_2020_paper.pdf)
# code [CSQ-pytorch](https://github.com/yuanli2333/Hadamard-Matrix-for-hashing)

# AlexNet
# [CSQ] epoch:65, bit:64, dataset:cifar10-1, MAP:0.787, Best MAP: 0.790
# [CSQ] epoch:90, bit:16, dataset:imagenet, MAP:0.593, Best MAP: 0.596, paper:0.601
# [CSQ] epoch:150, bit:64, dataset:imagenet, MAP:0.698, Best MAP: 0.706, paper:0.695
# [CSQ] epoch:40, bit:16, dataset:nuswide_21, MAP:0.784, Best MAP: 0.789
# [CSQ] epoch:40, bit:32, dataset:nuswide_21, MAP:0.821, Best MAP: 0.821
# [CSQ] epoch:40, bit:64, dataset:nuswide_21, MAP:0.834, Best MAP: 0.834

# ResNet50
# [CSQ] epoch:20, bit:64, dataset:imagenet, MAP:0.881, Best MAP: 0.881, paper:0.873
# [CSQ] epoch:10, bit:64, dataset:nuswide_21_m, MAP:0.844, Best MAP: 0.844, paper:0.839
# [CSQ] epoch:40, bit:64, dataset:coco, MAP:0.870, Best MAP: 0.883, paper:0.861

eps = 1e-8

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_est", help = 'Gradient Estimator for Discrete Latent', type = str, default = 'st')
    parser.add_argument("--u2g_samples", help = 'Number of u2g samples for CLS loss head', type = int, default = 1)
    #Assume u2g_recon < u2g_samples 
    # parser.add_argument("--u2g_recon", help = 'Number of u2g samples for reconstruction head', type = int, default = 1)
    parser.add_argument("--KL_regularization", help = 'scalar to tradeoff likelihood term and KL term', type = float, default = 0)
    parser.add_argument("--prior", help = 'prior Bernoulli success probability', type = float, default = 0.5)
    # parser.add_argument("--st_pretrain", help = 'Number of epochs pretrain with ST (Valid for U2G for now)', type = int, default = 0)
    #Todo: inplement pairwise dataset class for CIFAR-10 dataset
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--no-pairwise', dest='pairwise', action='store_false')
    parser.set_defaults(pairwise=False)
    parser.add_argument('--pair_weight', help = 'scalar to tradeoff pairwise term', type = float, default = 0)
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.set_defaults(tqdm=False)
    parser.add_argument('--dataset', type = str, default = "imagenet", help = " ")
    parser.add_argument('--rt_st', type = str, default = "ham", help = "Retrieval Stratege, \"ham\" or \"hamuct\"")
    parser.add_argument('--uct_quant_levels', type = int, nargs = '*', default = [2, 4], help = "Quantization levels for HashUQ-Bin (2 = 1 bit, 4 = 2 bits), evaluated alongside HashUQ")
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--net', dest = 'net_name', type = str, default = "alexnet", help = "alexnet, resnet18, resnet34, resnet50, ...")
    parser.add_argument('--bit', dest = 'bit_list', type = int, nargs = '+', default = [16], help = "e.g. --bit 16 32 64")
    parser.add_argument('--epoch', type = int, default = 100)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--no-val', dest='val', action='store_false')
    parser.add_argument("--train", action = 'store_true')
    parser.add_argument('--no-train', dest='train', action='store_false') 
    parser.set_defaults(train=True)       
    parser.add_argument('--sample', type = int, default = 1)
    parser.add_argument('--sample_method', type = str, default = "MCD") #MCD: MC Dropout, IGN: Input Gaussian Noise
    parser.add_argument('--eps', type = float, default = 1e-100)
    parser.set_defaults(val=False)    
    args = parser.parse_args()
    print('Arguments:', args)
    return args

def get_config():
    config = {
        "lambda": 0.0001,
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net": ResNet,
        # "dataset": "cifar10-1",
        # "dataset": "imagenet",
        # "dataset": "coco",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        "epoch": 100,
        "test_map": 10,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [16],
        "seed": 1, 
    }
    # config = config_dataset(config)
    return config


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.bit = bit
        self.is_single_label = config["dataset"] not in {"nuswide_21", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

    def forward(self, u, y, ind, config):
        # u = u.tanh()
        prob = torch.sigmoid(u)

        hash_center = self.label2center(y)
        # print("hash centre:", hash_center)
        # Hamming_loss = torch.sum(torch.abs(hash_center - b)/2, dim = (0, 1))/(config['batch_size']*bit)

        if config["grad_est"] == 'cf':
            Hamming_loss = torch.sum(prob + (hash_center+1)/2 - 2*prob*(hash_center+1)/2)/(config['batch_size']*self.bit)
        elif config["grad_est"] == 'cf2':
            Hamming_loss = torch.sum(prob*(1 - (hash_center+1)) + ((hash_center+1)/2)**2)/(config['batch_size']*self.bit)
        else:
            if config["grad_est"] == 'st':
                b = 2 * self.Sample_Bernoulli.apply(prob) - 1
            elif config["grad_est"] == 'gs':
                b = 2 * self.Sample_GS(prob, config) - 1
            elif config["grad_est"] == 'u2g':
                b = 2 * self.Sample_Bernoulli.apply(prob) - 1
            Hamming_loss = torch.sum(torch.abs(hash_center - b)/2, dim = (0, 1))/(config['batch_size']*self.bit)
        return Hamming_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    def is_power_of_two(self, n):
        if n <= 0:
            return False
        return (n & (n - 1)) == 0

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        if self.is_power_of_two(bit):
        #If bit = 2^n, ELSE: 
            H_K = hadamard(bit)
        else:
            # ceil_bit = int(math.pow(2,math.ceil(math.log2(bit))))
            # print("ceil bit", ceil_bit)
            # H_K = hadamard(ceil_bit)[:, random.sample(range(ceil_bit), bit)]
            floor_bit = int(math.pow(2,math.floor(math.log2(bit))))
            H_K = np.concatenate((hadamard(floor_bit), 2*np.random.randint(2, size = (floor_bit, bit - floor_bit)) - 1), axis = 1)

        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()
        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets

    def f_function_cls(self, b, y, config):
        hash_center = self.label2center(y)
        # center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))
        Hamming_loss = torch.sum(torch.abs(hash_center - b)/2, dim = (0, 1))/(config['batch_size']*self.bit)
        return Hamming_loss

    def estimate_grad(self, u, y, ind, config):
        batch_size, bit_length = u.size()
        sample_u = torch.rand((config["u2g_samples"], batch_size, bit_length)).to(config["device"])
        Score2 = torch.sigmoid(u)
        Score1 = 1 - Score2
        Indicator1 = (sample_u>Score1).float()
        Indicator2 = (sample_u<Score2).float()
        loss_1 = []
        loss_2 = []
        with torch.no_grad():
            for i in range(config["u2g_samples"]):
                loss_1_im = self.f_function_cls(2 * Indicator1[i] - 1, y, config)
                loss_2_im = self.f_function_cls(2 * Indicator2[i] - 1, y, config)
                loss_1.append(loss_1_im.item())
                loss_2.append(loss_2_im.item())
                del loss_1_im, loss_2_im
                torch.cuda.empty_cache()
            loss_1 = torch.as_tensor(loss_1).to(config["device"])
            loss_2 = torch.as_tensor(loss_2).to(config["device"])

        grad_u2g = torch.mean(torch.sigmoid(torch.abs(u))*(loss_1 - loss_2).unsqueeze(1).unsqueeze(2).repeat(1, batch_size, bit_length)*(Indicator1 - Indicator2)/2, dim = 0)
        return grad_u2g.detach()

    def estimate_loss(self, u, y, ind, config):
        prob = torch.sigmoid(u)
        b = 2 * self.Sample_Bernoulli.apply(prob) - 1
        hash_center = self.label2center(y)
        Hamming_loss = torch.sum(torch.abs(hash_center - b)/2, dim = (0, 1))/(config['batch_size']*self.bit)
        return Hamming_loss

    def Sample_GS(self, prob, config, lam = 0.5):
        size = prob.size()
        u = torch.rand(size).to(config['device'])
        alpha = (prob + eps)/(1 - prob + eps)
        L = torch.log(u) - torch.log(1 - u)
        bb = torch.sigmoid((torch.log(alpha) + L)/lam)
        return bb

    class Sample_Bernoulli(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            code = torch.bernoulli(input)
            return code

        @staticmethod
        def backward(ctx, grad_output):
            return grad_output

def train_val(config, bit):
    device = config["device"]

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    if not config["tqdm"]:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

    #Need to check whether we use save_path somewhere else.
    save_path = os.path.join(".", "saved_model", "BerVAE_bit_%i_optimizer_%s_KL_%.2f_dataset_%s_net_%s_seed_%i" %(bit, config["grad_est"], config["KL_regularization"], config["dataset"], config["net_name"], config["seed"]))
    print("save_path", save_path)
    config["save_path"] = save_path

    if not os.path.exists(os.path.join(".", "saved_model")):
        os.mkdir(os.path.join(".", "saved_model"))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    training_loss_list = []
    val_loss_list = []
    map_list = []
    kl_list = []
    pw_list = []

    torch.set_printoptions(sci_mode = True)

    if config["val"] == True:
        train_loader, val_loader, test_loader, dataset_loader, num_train, num_val, num_test, num_dataset = get_data(config)
    else:
        train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)


    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    # optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = CSQLoss(config, bit)

    optimizer_net = config["optimizer"]["type"](list(net.parameters()), **(config["optimizer"]["optim_params"]))
    # optimizer_criterion = config["optimizer"]["type"](list(criterion.parameters()), **(config["optimizer"]["optim_params"]))

    Best_mAP = 0

    save_hash_center(criterion, config)

    if config["train"] == True:
        for epoch in range(config["epoch"]):
            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

            print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

            net.train()

            train_loss = 0
            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device)

                # optimizer.zero_grad()
                optimizer_net.zero_grad()
                u = net(image)

                # print('u', u)

                prob = torch.sigmoid(u)
                loss = criterion(u, label.float(), ind, config)
                KL = torch.sum((prob*torch.log(torch.div(prob, config["prior"]) + eps) + (1 - prob)*torch.log(torch.div(1- prob, 1 - config["prior"]) + eps)))/(config['batch_size']*bit)
                KL_loss = config["KL_regularization"]*KL
                train_loss += loss.item()

                if config["grad_est"] in ['st', 'gs', 'cf', 'cf2']:
                    loss.backward(retain_graph = True)
                    KL_loss.backward(retain_graph = True)
                    optimizer_net.step()
                elif config["grad_est"] == 'u2g':
                    u_grad = criterion.estimate_grad(u, label.float(), ind, config)
                    u.backward(u_grad.detach(), retain_graph = True)
                    KL_loss.backward(retain_graph = True)
                    optimizer_net.step()

            train_loss = train_loss / len(train_loader)

            print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

            if (epoch + 1) % config["test_map"] == 0:
                if config["val"] == True:
                    Best_mAP, mAP = self_validate(config, Best_mAP, val_loader, net, bit, epoch, num_val, return_map = True)
                else:
                    Best_mAP, mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset, return_map = True)
                
                net.eval()
                criterion.eval()
                train_loss = 0
                with torch.no_grad():
                    for data in train_loader:
                        (image, label, ind) = data
                        image = image.to(device)
                        label = label.to(device)
                        u = net(image)
                        loss = criterion.estimate_loss(u, label.float(), ind, config)
                        train_loss += loss.item()
                    train_loss = train_loss / len(train_loader)

                val_loss = 0
                if config["val"] == True:
                    with torch.no_grad():
                        for data in val_loader:
                            (image, label, ind) = data
                            image = image.to(device)
                            label = label.to(device)
                            u = net(image)
                            loss = criterion.estimate_loss(u, label.float(), ind, config)
                            val_loss += loss.item()
                        val_loss = val_loss / len(val_loader)

                torch.save(net.state_dict(), os.path.join(save_path, "epoch_%i.pth" %(epoch)))

                training_loss_list.append(train_loss)
                val_loss_list.append(val_loss)
                map_list.append(mAP)
                kl_list.append(KL.item())
    else:
        map_list = pd.read_csv(os.path.join(save_path, "map.csv"))["0"]

    #Training Done    
    Best_Epoch = (np.argmax(map_list) + 1)*config["test_map"]
    Best_Model_Path = os.path.join(save_path, "epoch_%i.pth" %(Best_Epoch - 1))

    net.load_state_dict(torch.load(Best_Model_Path, map_location=device))
    Best_mAP = 0
    Best_mAP, mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, Best_Epoch - 1, num_dataset, return_map = True, tag = "final")

    if config["train"] == True:
        pd.DataFrame(training_loss_list).to_csv(os.path.join(save_path, "train_loss.csv"))
        pd.DataFrame(val_loss_list).to_csv(os.path.join(save_path, "val_loss.csv"))
        pd.DataFrame(map_list).to_csv(os.path.join(save_path, "map.csv"))
        pd.DataFrame(kl_list).to_csv(os.path.join(save_path, "kl.csv"))

if __name__ == "__main__":
    config = get_config()
    print(config)
    args = get_args()
    config.update(vars(args))
    config = config_dataset(config) 
    config["net"] = get_net(config["net_name"])
    # assert config["u2g_samples"]>=config["u2g_recon"]
    for bit in config["bit_list"]:
        if config["rt_st"] == 'ham':
            config["pr_curve_path"] = f"log/{config['net_name']}/CSQ_{config['dataset']}_{bit}_seed_{config['seed']}.json"
        train_val(config, bit)
