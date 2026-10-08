from utils.tools import *
from network import *

import os
import torch
import torch.optim as optim
import time
import numpy as np
import random
import argparse
import pandas as pd
from functools import partialmethod


from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy('file_system')


# GreedyHash(NIPS2018)
# paper [Greedy Hash: Towards Fast Optimization for Accurate Hash Coding in CNN](https://papers.nips.cc/paper/7360-greedy-hash-towards-fast-optimization-for-accurate-hash-coding-in-cnn.pdf)
# code [GreedyHash](https://github.com/ssppp/GreedyHash)

eps = 1e-8


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad_est", help = 'Gradient Estimator for Discrete Latent', type = str, default = 'st')
    parser.add_argument("--u2g_samples", help = 'Number of u2g samples', type = int, default = 100)
    parser.add_argument("--KL_regularization", help = 'scalar to tradeoff likelihood term and KL term', type = float, default = 0)
    parser.add_argument("--prior", help = 'prior Bernoulli success probability', type = float, default = 0.5)
    parser.add_argument("--st_pretrain", help = 'Number of epochs pretrain with ST (Valid for U2G for now)', type = int, default = 0)
    parser.add_argument('--pairwise', action='store_true')
    parser.add_argument('--no-pairwise', dest='pairwise', action='store_false')
    parser.set_defaults(pairwise=False)
    parser.add_argument('--dataset', type = str, default="imagenet", help = " ")
    parser.add_argument('--rt_st', type = str, default = "ham", help = "Retrieval Stratege, \"ham\" or \"hamuct\"")
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--no-val', dest='val', action='store_false')
    parser.add_argument("--train", action = 'store_true')
    parser.add_argument('--no-train', dest='train', action='store_false') 
    parser.set_defaults(train=True)       
    parser.add_argument('--sample', type = int, default = 1)
    parser.add_argument('--sample_method', type = str, default = "MCD") #MCD: MC Dropout, IGN: Input Gaussian Noise

    parser.set_defaults(val=False)    
    parser.add_argument('--tqdm', action='store_true')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false')
    parser.add_argument('--uct_quant_levels', type = int, nargs = '*', default = [2, 4], help = "Quantization levels for HashUQ-Bin (2 = 1 bit, 4 = 2 bits), evaluated alongside HashUQ")
    parser.add_argument('--seed', type = int, default = 0)
    parser.add_argument('--net', dest = 'net_name', type = str, default = "alexnet", help = "alexnet, resnet18, resnet34, resnet50, ...")
    parser.add_argument('--bit', dest = 'bit_list', type = int, nargs = '+', default = [16], help = "e.g. --bit 16 32 64")
    parser.add_argument('--epoch', type = int, default = 100)
    parser.add_argument('--num_workers', type = int, default = 4)
    parser.add_argument('--eps', type = float, default = 1e-100)
    parser.set_defaults(tqdm=False)

    args = parser.parse_args()
    print('Arguments:', args)
    return args

def get_config():
    config = {
        "alpha": 0.1,
        "optimizer": {"type": optim.SGD, "epoch_lr_decrease": 30,
                      "optim_params": {"lr": 0.001, "weight_decay": 5e-4, "momentum": 0.9}},
        # "optimizer": {"type": optim.Adam, "epoch_lr_decrease": 30,
        #               "optim_params": {"lr": 0.001, "weight_decay": 5e-4}},
        # "optimizer": {"type": optim.RMSprop, "epoch_lr_decrease": 30,
        #               "optim_params": {"lr": 5e-5, "weight_decay": 5e-4}},

        "info": "[GreedyHash]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": AlexNet,
        # "net":ResNet,
        # "dataset": "cifar10",
        # "dataset": "cifar10-1",
        # "dataset": "cifar10-2",
        # "dataset": "coco",
        # "dataset": "mirflickr",
        # "dataset": "voc2012",
        "dataset": "imagenet",
        # "dataset": "nuswide_21",
        # "dataset": "nuswide_21_m",
        # "dataset": "nuswide_81_m",
        "epoch": 100,
        "test_map": 100,
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [16],
        "seed": 0, 
        # "grad_est": "u2g" #gs, u2g, st, debug
    }
    # config = config_dataset(config)

    return config


class GreedyHashLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(GreedyHashLoss, self).__init__()
        self.fc = torch.nn.Linear(bit, config["n_class"], bias=False).to(config["device"])
        
        self.criterion = torch.nn.CrossEntropyLoss().to(config["device"])
        self.Sigmoid = torch.nn.Sigmoid()

    def forward(self, u, onehot_y, ind, config):
        #Sigmoid layer
        # u = 2*self.Sigmoid(u) - 1
        prob = torch.sigmoid(u)
        if config["grad_est"] == 'st':
            b = 2 * GreedyHashLoss.Sample_Bernoulli.apply(prob) - 1
        elif config["grad_est"] == 'gs':
            b = 2 * self.Sample_GS(prob, config) - 1
        elif config["grad_est"] == 'u2g':
            b = 2 * GreedyHashLoss.Sample_Bernoulli.apply(prob) - 1
        elif config["grad_est"] == 'debug':
            print('Prob:', prob)
            b = 2 * self.Sample_GS(prob, config) - 1

        # one-hot to label
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        # loss2 = config["alpha"] * (u.abs() - 1).pow(3).abs().mean()
        # print('loss_1', loss1)
        return loss1

    def f_function(self, b, onehot_y):
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        # print('loss_2', loss1)
        return loss1

    def estimate_grad(self, u, onehot_y, ind, config):

        batch_size, bit_length = u.size()
        #num_sample,batch_size, bit_length
        sample_u = torch.rand((config["u2g_samples"], batch_size, bit_length)).to(config["device"])
        # print('unif rv',sample_u.size())
        Score2 = torch.sigmoid(u)
        Score1 = 1 - Score2
        # print('Score 1', Score1.size())
        # print('Score 2', Score2.size())
        Indicator1 = (sample_u>Score1).float()
        Indicator2 = (sample_u<Score2).float()
        # print('Ind 1',Indicator1.size())
        # print('Ind 2',Indicator2.size())
        loss_1 = []
        loss_2 = []
        for i in range(config["u2g_samples"]):
            loss_1.append(self.f_function(2 * Indicator1[i] - 1, onehot_y))
            loss_2.append(self.f_function(2 * Indicator2[i] - 1, onehot_y))
        loss_1 = torch.as_tensor(loss_1).to(config["device"])
        loss_2 = torch.as_tensor(loss_2).to(config["device"])
        # print('loss_1', loss_1.size())
        # print('loss 2', loss_2.size())
        # print('loss2 - loss1', (loss_1 - loss_2).unsqueeze(1).unsqueeze(2).repeat(1, batch_size, bit_length).size())
        # print('loss_1', loss_1.size())
        # print('loss_2', loss_2.size())
        grad_u2g = torch.mean(torch.sigmoid(torch.abs(u))*(loss_1 - loss_2).unsqueeze(1).unsqueeze(2).repeat(1, batch_size, bit_length)*(Indicator1 - Indicator2)/2, dim = 0)
        # grad_arm = (loss_1 - loss_2).unsqueeze(0).unsqueeze(1).repeat(batch_size, bit_length)*(sample_u - 1/2)
        # print('Grad U2G', grad_u2g.size())
        # print('Grad ARM', grad_arm.size())
        return grad_u2g.detach()

    def Sample_GS(self, prob, config, lam = 0.5):
        size = prob.size()
        u = torch.rand(size).to(config['device'])
        alpha = (prob + eps)/(1 - prob + eps)
        L = torch.log(u) - torch.log(1 - u)
        bb = torch.sigmoid((torch.log(alpha) + L)/lam)
        return bb

    def estimate_loss(self, u, onehot_y, ind, config):
        #Estimate the cls loss by sampling the latent
        prob = torch.sigmoid(u)
        b = 2 * GreedyHashLoss.Sample_Bernoulli.apply(prob) - 1
        y = onehot_y.argmax(axis=1)
        y_pre = self.fc(b)
        loss1 = self.criterion(y_pre, y)
        return loss1

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            # ctx.save_for_backward(input)
            return input.sign()

        @staticmethod
        def backward(ctx, grad_output):
            # input,  = ctx.saved_tensors
            # grad_output = grad_output.data
            return grad_output

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

    save_path = os.path.join(".", "saved_model", "BerVAE_Greedy_bit_%i_dataset_%s_opt_%s_u2gsample_%i_KL_%.2f_net_%s_seed_%i" %(bit, config["dataset"], config["grad_est"], config["u2g_samples"], config["KL_regularization"], config["net_name"], config["seed"]))
    print("save_path", save_path)
    config["save_path"] = save_path
    if not os.path.exists(os.path.join(".", "saved_model")):
        os.mkdir(os.path.join(".", "saved_model"))

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    training_loss_list = []
    val_loss_list = []
    map_list = []

    torch.set_printoptions(sci_mode = True)

    # train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    if config["val"] == True:
        train_loader, val_loader, test_loader, dataset_loader, num_train, num_val, num_test, num_dataset = get_data(config)
    else:
        train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)

    config["num_train"] = num_train
    net = config["net"](bit).to(device)

    criterion = GreedyHashLoss(config, bit)

    optimizer_net = config["optimizer"]["type"](list(net.parameters()), **(config["optimizer"]["optim_params"]))
    optimizer_criterion = config["optimizer"]["type"](list(criterion.parameters()), **(config["optimizer"]["optim_params"]))
    

    Best_mAP = 0

    if config["train"] == True:
        for epoch in range(config["epoch"]):

            lr = config["optimizer"]["optim_params"]["lr"] * (0.1 ** (epoch // config["optimizer"]["epoch_lr_decrease"]))
            for param_group in optimizer_net.param_groups:
                param_group['lr'] = lr
            for param_group in optimizer_criterion.param_groups:
                param_group['lr'] = lr

            current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

            print("%s[%2d/%2d][%s] bit:%d, lr:%.9f, dataset:%s, training...." % (
                config["info"], epoch + 1, config["epoch"], current_time, bit, lr, config["dataset"]), end="")

            net.train()

            train_loss = 0
            for image, label, ind in train_loader:
                image = image.to(device)
                label = label.to(device)

                optimizer_net.zero_grad()
                optimizer_criterion.zero_grad()

                u = net(image)

                prob = torch.sigmoid(u)
                # if epoch > 5:
                #     print('Bernoulli Prob', prob)

                loss = criterion(u, label.float(), ind, config)
                
                KL = torch.sum((prob*torch.log(torch.div(prob, config["prior"]) + eps) + (1 - prob)*torch.log(torch.div(1- prob, 1 - config["prior"]) + eps)))/(config['batch_size']*bit)
                KL_loss = config["KL_regularization"]*KL
                # print('KL', KL)
                train_loss += loss.item()


                if config["grad_est"] in ['st', 'gs']:
                    loss.backward(retain_graph = True)
                    KL_loss.backward()
                    optimizer_criterion.step()
                    optimizer_net.step()
                elif config["grad_est"] == 'u2g':
                    # print("u_grad:", u_grad)

                    if epoch < config["st_pretrain"]:
                        loss.backward(retain_graph = True)
                        KL_loss.backward()
                        optimizer_criterion.step()
                        optimizer_net.step()
                    else:
                        loss.backward(retain_graph = True)
                        optimizer_criterion.step()
                        optimizer_net.zero_grad()
                        optimizer_criterion.zero_grad()
                        # u_grad = criterion.estimate_grad(u, label.float(), ind, config)

                        u_grad = []
                        # for i in range(config["u2g_samples"]):
                        #     u_grad.append(criterion.estimate_grad(u, label.float(), ind, config))
                        # u_grad_avg = sum(u_grad)/len(u_grad)
                        u_grad = criterion.estimate_grad(u, label.float(), ind, config)

                        u.backward(u_grad.detach(), retain_graph = True)
                        KL_loss.backward()
                        optimizer_net.step()


                elif config["grad_est"] == 'debug':
                    u.retain_grad() #Save the gradient of u
                    u_grad = []
                    for i in range(100):
                        u_grad.append(criterion.estimate_grad(u, label.float(), ind, config))
                    u_grad_avg = sum(u_grad)/len(u_grad)

                    loss.backward()
                    print("u2g_grad:", u_grad_avg)
                    print("gs grad:", u.grad)
                    print("cosine between gs and u2g:", torch.nn.functional.cosine_similarity(u_grad_avg, u.grad, dim = 1))
                    optimizer_criterion.step()
                    optimizer_net.step()

                
            train_loss = train_loss / len(train_loader)

            print("\b\b\b\b\b\b\b loss:%.3f KL:%.3e" % (train_loss, KL))

            if (epoch + 1) % config["test_map"] == 0:
                if config["val"] == True:
                    Best_mAP, mAP = self_validate(config, Best_mAP, val_loader, net, bit, epoch, num_val, return_map = True)
                else:
                    Best_mAP, mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset, return_map = True)

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
                val_loss_list.append(val_loss)
                map_list.append(mAP)

                # net.eval()
                # train_loss = 0
                # for image, label, ind in train_loader:
                #     image = image.to(device)
                #     label = label.to(device)
                #     u = net(image)
                #     loss = criterion.estimate_loss(u, label.float(), ind, config)
                #     train_loss += loss.item()
                # train_loss = train_loss / len(train_loader)
                # training_loss_list.append(train_loss)
                # map_list.append(mAP)
                # net.train()
        
        #Save Model and Hash-Code
    else:
        map_list = pd.read_csv(os.path.join(save_path, "map.csv"))["0"]
    Best_Epoch = (np.argmax(map_list) + 1)*config["test_map"]
    Best_Model_Path = os.path.join(save_path, "epoch_%i.pth" %(Best_Epoch - 1))

    net.load_state_dict(torch.load(Best_Model_Path, map_location=device))
    Best_mAP = 0
    Best_mAP, mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, Best_Epoch - 1, num_dataset, return_map = True, tag = "final")
    if config["train"] == True:
        # pd.DataFrame(training_loss_list).to_csv(os.path.join(save_path, "train_loss.csv"))
        pd.DataFrame(map_list).to_csv(os.path.join(save_path, "map.csv"))

        pd.DataFrame(training_loss_list).to_csv(os.path.join("./","val_loss_bit_%i_optimizer_%s_u2gsample_%i.csv" %(bit, config["grad_est"], config["u2g_samples"])))
        pd.DataFrame(map_list).to_csv(os.path.join("./", "map_bit_%i_optimizer_%s_u2gsample_%i.csv"%(bit, config["grad_est"], config["u2g_samples"])))


if __name__ == "__main__":
    config = get_config()
    print(config)
    args = get_args()
    config.update(vars(args))
    config = config_dataset(config) 
    config["net"] = get_net(config["net_name"])
    if config["dataset"] == "imagenet":
        config["alpha"] = 1
        config["optimizer"]["epoch_lr_decrease"] = 80
    for bit in config["bit_list"]:
        train_val(config, bit)
