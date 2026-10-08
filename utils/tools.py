import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import torchvision.datasets as dsets
import os
import json
import random 
from torch.distributions import Bernoulli
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import scipy
import matplotlib.pyplot as plt

def config_dataset(config):
    if "cifar" in config["dataset"]:
        config["topK"] = -1
        config["n_class"] = 10
    elif config["dataset"] in ["nuswide_21", "nuswide_21_m"]:
        config["topK"] = 5000
        config["n_class"] = 21
    elif config["dataset"] == "nuswide_81_m":
        config["topK"] = 5000
        config["n_class"] = 81
    elif config["dataset"] == "coco":
        config["topK"] = 5000
        config["n_class"] = 80
    elif config["dataset"] == "imagenet":
        config["topK"] = 1000
        config["n_class"] = 100
    elif config["dataset"] == "mirflickr":
        config["topK"] = -1
        config["n_class"] = 38
    elif config["dataset"] == "voc2012":
        config["topK"] = -1
        config["n_class"] = 20

    config["data_path"] = "../image_hashing_data/" + config["dataset"] + "/"
    if config["dataset"] == "nuswide_21":
        config["data_path"] = "../image_hashing_data/nuswide_21/"
    if config["dataset"] in ["nuswide_21_m", "nuswide_81_m"]:
        config["data_path"] = "../image_hashing_data/nus_wide_m/"
    if config["dataset"] == "coco":
        config["data_path"] = "../image_hashing_data/coco/"
    if config["dataset"] == "voc2012":
        config["data_path"] = "../image_hashing_data/voc2012/"

    if config['val'] == True:
        config["data"] = {
            "train_set": {"list_path": "../image_hashing_data/" + config["dataset"] + "/new_train.txt", "batch_size": config["batch_size"]},
            "val": {"list_path": "../image_hashing_data/" + config["dataset"] + "/val.txt", "batch_size": config["batch_size"]},
            "database": {"list_path": "../image_hashing_data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": "../image_hashing_data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    else:
        config["data"] = {
            "train_set": {"list_path": "../image_hashing_data/" + config["dataset"] + "/train.txt", "batch_size": config["batch_size"]},
            "database": {"list_path": "../image_hashing_data/" + config["dataset"] + "/database.txt", "batch_size": config["batch_size"]},
            "test": {"list_path": "../image_hashing_data/" + config["dataset"] + "/test.txt", "batch_size": config["batch_size"]}}
    return config

class ImageList(torch.utils.data.Dataset):
    def __init__(self, data_path, image_list, transform):
        self.imgs = [(os.path.join(data_path, val.split()[0]), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)

class ImageList_Pair(torch.utils.data.Dataset):
    def __init__(self, data_path, image_list, transform, prob = 0.5, idx = None):
        #Positive pair probability
        self.imgs = [(os.path.join(data_path, val.split()[0]), np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        if idx is None:
            self.sim_matrix = np.load(os.path.join(data_path, "train.npy"))
        else:
            self.sim_matrix = np.load(os.path.join(data_path, "train.npy"))[idx][:, idx]
        self.transform = transform
        self.prob = prob
        self.labels = [i[1] for i in self.imgs]
        
    def sample_similar_data(self, index1):
        # sim_data = np.array([np.array_equal(i, self.labels[index1]) for i in self.labels])
        # sim_idx = np.argwhere(sim_data == 1)[:, 0]
        sim_idx = np.argwhere(self.sim_matrix[index1] == 1)[:,0]
        index = random.choice(sim_idx)
        return index
    
    def sample_diff_data(self, index1):
        # sim_data = np.array([np.array_equal(i, self.labels[index1]) for i in self.labels])
        # unsim_idx = np.argwhere(sim_data == 0)[:, 0]
        unsim_idx = np.argwhere(self.sim_matrix[index1] == 0)[:,0]
        index = random.choice(unsim_idx)
        return index

    def __getitem__(self, index1):
        #Return img1, img2, target1, target2, index1, index2, sim
        path1, target1 = self.imgs[index1]
        img1 = Image.open(path1).convert('RGB')
        img1 = self.transform(img1)
        
        randnum = np.random.random()

        if randnum > self.prob: #img1 and img2 belongs to different category
            sim = -1
            index2 = self.sample_diff_data(index1)
            #Sample img 2
            
        else:# img1 and img2 belongs to the same category
            sim = 1
            index2 = self.sample_similar_data(index1)
            #Sample img 2
        
        path1, target1 = self.imgs[index1]
        img1 = Image.open(path1).convert('RGB')
        img1 = self.transform(img1)
        path2, target2 = self.imgs[index2]
        img2 = Image.open(path2).convert('RGB')
        img2 = self.transform(img2)

        return img1, target1, index1, img2, target2, index2, sim
    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, crop_size, data_set):
    if data_set == "train_set":
        step = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(crop_size)]
    else:
        step = [transforms.CenterCrop(crop_size)]
    return transforms.Compose([transforms.Resize(resize_size)]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(config):
    batch_size = config["batch_size"]

    train_size = 500
    test_size = 100

    np.random.seed(seed = config["seed"])

    if config["dataset"] == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(config["crop_size"]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/dataset/cifar/'
    # Dataset
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if config["dataset"] == "cifar10":
        # test:1000, train:5000, database:54000
        pass
    elif config["dataset"] == "cifar10-1":
        # test:1000, train:5000, database:59000
        database_index = np.concatenate((train_index, database_index))
    elif config["dataset"] == "cifar10-2":
        # test:10000, train:50000, database:50000
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(config["seed"])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=config.get("num_workers", 4),
                                               worker_init_fn=seed_worker,
                                               generator=g,)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=config.get("num_workers", 4), 
                                              worker_init_fn = seed_worker, 
                                              generator = g,)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=1024,
                                                  shuffle=False,
                                                  num_workers=config.get("num_workers", 4), 
                                                  worker_init_fn = seed_worker, 
                                                  generator = g,)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(config):
    if "cifar" in config["dataset"]:
        return cifar_dataset(config)

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    if config["val"] == True:
        split_list = ["train_set", "val", "test", "database"]
    else:
        split_list = ["train_set", "test", "database"]

    for data_set in split_list:
        if (config["pairwise"] == True) and (data_set == "train_set"):
            dsets[data_set] = ImageList_Pair(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        else:
            dsets[data_set] = ImageList(config["data_path"],
                                        open(data_config[data_set]["list_path"]).readlines(),
                                        transform=image_transform(config["resize_size"], config["crop_size"], data_set))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle= (data_set == "train_set") , num_workers=config.get("num_workers", 4))
    if config["val"] == True:
        return dset_loaders["train_set"], dset_loaders["val"] ,dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["val"]), len(dsets["test"]), len(dsets["database"])
    else:
        return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
            len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def get_train_list(config):
    train_datalist = os.path.join(config["data_path"], "train.txt")
    train_img = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in open(train_datalist).readlines()]
    return train_img

def build_dataloader(datalist, index, config, data_set, pair = True):
    if pair == True:
        dataset =  ImageList_Pair(config["data_path"], datalist, transform=image_transform(config["resize_size"], config["crop_size"], data_set), idx = index)
    else:
        dataset =  ImageList(config["data_path"], datalist, transform=image_transform(config["resize_size"], config["crop_size"], data_set))
    if data_set == 'train_set':    
        return torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config["batch_size"],
                                                shuffle=True,
                                                num_workers=config.get("num_workers", 4))
    else:
        return torch.utils.data.DataLoader(dataset=dataset,
                                                batch_size=config["batch_size"],
                                                shuffle=False,
                                                num_workers=config.get("num_workers", 4))        

def construct_datalist(train_list, train_idx, pool_list, pool_idx, config, move_idx = None, img_list = None):
    #If idx is not none, move the uncertain data
    #If the training list is empty, construct the training list
    total_num_cls = config["n_class"]
    #Todo
    num_sample_per_cls = 2
    if len(train_list) == 0:
        training_set_path = open(os.path.join(config["data_path"], "train.txt")).readlines()
        labels = [np.where(i[1] == 1)[0][0] for i in img_list]
        index_class = []
        [index_class.append([]) for i in range(total_num_cls)]
        for idx, i in enumerate(img_list):
            index_class[labels[idx]].append(idx)
        Sample_idx = []
        [Sample_idx.append(random.sample(i, config["init_spl"]//total_num_cls)) for i in index_class]
        Sample_idx = np.array(Sample_idx).reshape(-1)
        Pool_idx = list(set(range(len(img_list))).difference(Sample_idx))
        sample_img = []
        [sample_img.append(training_set_path[i]) for i in Sample_idx]
        Pool_img = []
        [Pool_img.append(training_set_path[i]) for i in Pool_idx]
        return sample_img, Sample_idx, Pool_img, Pool_idx
    else:
        if move_idx is None:
            idx = random.sample(range(len(pool_idx)), config["spl_iter"])
        else:
            idx = move_idx
        #Sort idx
        # print(idx)
        list.sort(idx, reverse = True)
        # print(idx)
        [train_list.append(pool_list[i]) for i in idx]
        [train_idx.append(pool_idx[i]) for i in idx]
        [pool_list.pop(i) for i in idx]
        [pool_idx.pop(i) for i in idx]
        return train_list, train_idx, pool_list, pool_idx

# def get_most_uncertain(pool_dataloader, model, config):
#     #Return the index of most uncertain data samples
#     pool_binary, pool_label, pool_uct, pool_prob = compute_result(pool_dataloader, model, device="cuda", uct = True)
#     # print("largest uncertain", pool_uct[np.argsort(pool_uct.cpu().numpy())[-10:]])
#     # print("smallest uncertain", pool_uct[np.argsort(pool_uct.cpu().numpy())[:10]])

#     tst_prob_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(pool_prob)
#     uct_prob_embedded = tst_prob_embedded[np.argsort(pool_uct.cpu().numpy())[-config["spl_iter"]:]]
#     plt.scatter(tst_prob_embedded[:, 0], tst_prob_embedded[:, 1], color = 'b')
#     plt.scatter(uct_prob_embedded[:, 0], uct_prob_embedded[:, 1], color = 'r')
#     plt.savefig("visulize_sample.svg")
#     plt.show()
#     return np.argsort(pool_uct.cpu().numpy())[-config["spl_iter"]:]

def visulize_result(save_path, prob_embeded, uct_idx, pool_label):
    uct_prob_embedded = prob_embeded[uct_idx]
    plt.clf()
    plt.scatter(prob_embeded[:, 0], prob_embeded[:, 1], color = 'b')
    plt.scatter(uct_prob_embedded[:, 0], uct_prob_embedded[:, 1], color = 'r', alpha = 0.5)
    plt.savefig(save_path + '_tsne.svg')
    plt.show()
    plt.clf()
    plt.hist(np.argwhere(pool_label[uct_idx] == 1)[1], bins = range(101))
    plt.savefig(save_path + '_hist.svg')
    plt.show()    

def tsne_embedding(pool_prob):
    tst_prob_embedded = TSNE(n_components=2, init='random', perplexity=3).fit_transform(pool_prob)
    return tst_prob_embedded

def get_most_uncertain(pool_uct, config):
    return np.argsort(pool_uct.cpu().numpy())[-config["spl_iter"]:]

def get_ramdon_idx(pool_idx, config):
    return random.sample(range(len(pool_idx)), config["spl_iter"])

def save_hash_center(creterion, config):
    targets = creterion.hash_targets.cpu().numpy()
    path = os.path.join(config["save_path"], "hash_target.npy")
    np.save(path, targets)

def enable_mc_dropout(net):
    # MC Dropout: only Dropout layers sample, BatchNorm etc. stay in eval mode
    net.eval()
    n_dropout = 0
    for m in net.modules():
        if isinstance(m, torch.nn.modules.dropout._DropoutNd):
            m.train()
            n_dropout += 1
    if n_dropout == 0:
        print("WARNING: MC Dropout requested but the network has no Dropout layer, all samples will be identical")

def compute_result(dataloader, net, config, device, uct = False):
    bs, clses, entropy, p_value, log_p_value, var, act_var, prob = [], [], [], [], [], [], [], []
    if config["sample"] > 1 and config["sample_method"] == "MCD":
        enable_mc_dropout(net)
    else:
        net.eval()
    with torch.no_grad():
        for img, cls, _ in tqdm(dataloader):
        # for img, cls, _ in dataloader:
            clses.append(cls)
            prob_samples = []
            entropy_samples = []
            activ_samples = []
            for i in range(config["sample"]):
                if config["sample_method"] == "MCD":
                    activ_one_sample = net(img.to(device))
                elif config["sample_method"] == "IGN":
                    # print("image shape", img.shape)
                    gaussian_noise =  torch.randn(img.size(), device=device)*0.01
                    activ_one_sample = net(img.to(device) + gaussian_noise)
                prob_one_sample = torch.sigmoid(activ_one_sample).data.cpu()
                # map_prob = torch.abs(prob_one_sample - 0.5) + 0.5
                # min_idx = torch.argmin(torch.abs(prob_one_sample - 0.5), axis = 1)
                # sec_prob = torch.abs(prob_one_sample - 0.5)*(-2*torch.nn.functional.one_hot(min_idx, num_classes = prob_one_sample.size()[1]) + 1) + 0.5
                activ_samples.append(activ_one_sample)
                prob_samples.append(prob_one_sample)
                entropy_samples.append(torch.sum(Bernoulli(probs = prob_one_sample).entropy(), axis = 1))
            prob_samples = torch.stack(prob_samples)
            map_prob = torch.abs(prob_samples - 0.5).numpy() + 0.5
            min_idx = torch.argmin(torch.abs(prob_samples - 0.5), axis = 2)
            sec_prob = (torch.abs(prob_samples - 0.5)*(-2*torch.nn.functional.one_hot(min_idx, num_classes = prob_one_sample.size()[1]) + 1)).numpy() + 0.5
            least_prob = 0.5 - torch.abs(prob_samples - 0.5).numpy()
            # print(map_prob.shape, least_prob.shape)
            # tstat, pvalue =  scipy.stats.ttest_rel(np.prod(map_prob, axis = 2), np.prod(least_prob, axis = 2), axis = 0)
            if config["sample"] > 1:
                tstat, pvalue =  scipy.stats.ttest_rel(prob_samples.numpy(), (1 - prob_samples.numpy()), axis = 0)
                # tstat2, pvalue2 =  scipy.stats.ttest_rel(np.prod(map_prob, axis = 2), np.prod(sec_prob, axis = 2), axis = 0)
                prob_var = torch.mean(torch.var(prob_samples, dim = 0), dim = 1)
            else:
                pvalue = torch.zeros((entropy_samples[0].shape[0],1))
                prob_var = torch.zeros((entropy_samples[0].shape[0]))
            # print(pvalue.shape)
            prob_avg = torch.mean(prob_samples, dim = 0)
            activ_var = torch.mean(torch.var(torch.stack(activ_samples), dim = 0), dim = 1)
            # print("prob_var", prob_var)
            # print("prob_mean", prob_avg)
            # print("activ_var", activ_var)
            # if config["sample"] > 1:
            entropy_avg = sum(entropy_samples)/len(entropy_samples)
            bs.append(2*prob_avg - 1)
            if uct == True:
                entropy.append(entropy_avg)
                var.append(prob_var)
                p_value.append(torch.sum(((torch.tensor(pvalue))), dim = 1))
                log_p_value.append(torch.sum(torch.log((torch.tensor(pvalue)) + config["eps"]), dim = 1))
                # log_p_value.append(torch.sum(torch.log((torch.tensor(pvalue))), dim = 1))
                act_var.append(activ_var)
                prob.append(prob_avg)
    # print("entropy", entropy)
    # print("torch.cat(entropy)/torch.max(torch.cat(entropy))", (torch.cat(entropy)/torch.max(torch.cat(entropy))).shape)
    # print("torch.cat(var)/torch.max(torch.cat(var))",(torch.cat(var)/torch.max(torch.cat(var))).shape)
    # print("torch.cat(log_p_value)/torch.abs(torch.min(torch.cat(log_p_value)))", (torch.cat(log_p_value)/torch.abs(torch.min(torch.cat(log_p_value)))).shape)
    # if uct == True:
    #     print("torch.cat(log_p_value)", torch.cat(log_p_value))
    #     print("torch.cat(log_p_value).dtype", torch.cat(log_p_value).dtype)
    #     print("min pvalue", torch.min(torch.cat(log_p_value)))
    #     print("log_p_value", torch.cat(log_p_value)/torch.abs(torch.min(torch.cat(log_p_value))))
    if uct == True:
        return torch.cat(bs).sign(), torch.cat(clses), [torch.cat(log_p_value)/torch.abs(torch.min(torch.cat(log_p_value)))], torch.cat(prob)
    else:
        return torch.cat(bs).sign(), torch.cat(clses)

def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
    # for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        # print("hamm:", hamm.shape)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def self_CalcTopMap(rB, retrievalL, topk):
    # num_query = queryL.shape[0]
    num_query = rB.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
    # for iter in range(num_query):
        gnd = (np.dot(retrievalL[iter, :], np.delete(retrievalL, iter, axis = 0).transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(rB[iter, :], np.delete(rB, iter, axis = 0))
        # print("hamm:", hamm.shape)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

def find_quantile(entropy, equal = True, quant_level = 64):
    quant = []
    if equal == True:
        for i in range(quant_level+1):
            quant.append(min(entropy.reshape(-1)) + i*(max(entropy.reshape(-1)) - min(entropy.reshape(-1)))/quant_level)
    else:
        for i in range(quant_level+1):
            quant.append(np.quantile(entropy.reshape(-1), i/quant_level))
    return quant

def quantize_data(entropy, quantile):
    return np.array(quantile)[np.digitize(entropy, quantile,  right = True)]

def CalcTopMap_Uct(rB, qB, retrievalL, queryL, topk, r_Uct):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
    # for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        # print("hamm:", hamm.shape)
        ind = np.argsort(hamm + r_Uct)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

# Accumulates precision/recall sums over queries, so memory is O(num_gallery)
# instead of O(num_query * num_gallery)
def CalcTopMapWithPR(qB, queryL, rB, retrievalL, topk):
    num_query = queryL.shape[0]
    num_gallery = retrievalL.shape[0]
    topkmap = 0
    prec_total = np.zeros(num_gallery)
    recall_total = np.zeros(num_gallery)
    num_valid = 0
    return_images = np.arange(1, num_gallery + 1)
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)
        all_sim_num = np.sum(gnd)

        prec_sum = np.cumsum(gnd)
        prec_total += prec_sum / return_images
        recall_total += prec_sum / all_sim_num
        num_valid += 1

        assert all_sim_num == prec_sum[-1]

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    cum_prec = prec_total / max(num_valid, 1)
    cum_recall = recall_total / max(num_valid, 1)

    return topkmap, cum_prec, cum_recall

# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset, return_map = False, tag = None):
    device = config["device"]
    tst_binary, tst_label, tst_uct, tst_prob = compute_result(test_loader, net, config, device=device, uct = True)

    trn_uct = None
    if config["rt_st"] == "ham":
        trn_binary, trn_label = compute_result(dataset_loader, net, config, device=device)
    elif config["rt_st"] == "hamuct":
        trn_binary, trn_label, trn_uct, trn_prob = compute_result(dataset_loader, net, config, device=device, uct = True)
        trn_uct = trn_uct[0].detach().cpu().numpy()

    rB, qB = trn_binary.numpy(), tst_binary.numpy()
    rL, qL = trn_label.numpy(), tst_label.numpy()

    if "pr_curve_path" not in config:
        mAP = CalcTopMap(rB, qB, rL, qL, config["topK"])
    else:
        mAP, cum_prec, cum_recall = CalcTopMapWithPR(qB, qL, rB, rL, config["topK"])
        index_range = num_dataset // 100
        index = [i * 100 - 1 for i in range(1, index_range + 1)]
        max_index = max(index)
        overflow = num_dataset - index_range * 100
        index = index + [max_index + i for i in range(1, overflow + 1)]
        c_prec = cum_prec[index]
        c_recall = cum_recall[index]

        pr_data = {
            "index": index,
            "P": c_prec.tolist(),
            "R": c_recall.tolist()
        }
        os.makedirs(os.path.dirname(config["pr_curve_path"]), exist_ok=True)
        with open(config["pr_curve_path"], 'w') as f:
            f.write(json.dumps(pr_data))
        print("pr curve save to ", config["pr_curve_path"])
    print("mAP_No_Uct", mAP)

    results = {"dataset": config["dataset"], "net": config.get("net_name"), "bit": int(bit), "epoch": int(epoch) + 1,
               "seed": config["seed"], "sample": config["sample"], "sample_method": config["sample_method"],
               "topK": config["topK"], "mAP_Hamming": float(mAP)}

    # HashUQ: rank by Hamming distance, break ties with database uncertainty
    if trn_uct is not None:
        mAP_uct = CalcTopMap_Uct(rB, qB, rL, qL, config["topK"], trn_uct)
        results["mAP_HashUQ"] = float(mAP_uct)
        print("mAP_HashUQ", mAP_uct)
        # HashUQ-Bin: uncertainty quantized into `level` quantile bins before ranking
        for level in config.get("uct_quant_levels", []):
            quantile = find_quantile(trn_uct, equal = False, quant_level = level)
            mAP_bin = CalcTopMap_Uct(rB, qB, rL, qL, config["topK"], quantize_data(trn_uct, quantile))
            results["mAP_HashUQ-Bin_L%i" % level] = float(mAP_bin)
            print("mAP_HashUQ-Bin (quant_level=%i)" % level, mAP_bin)

    if "save_path" in config:
        result_file = os.path.join(config["save_path"], "results_%s.json" % (tag if tag is not None else "epoch_%i" % (epoch + 1)))
        with open(result_file, "w") as f:
            json.dump(results, f, indent = 2)
        print("results saved to", result_file)

    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            np.save(os.path.join(save_path, "tst_label.npy"), qL)
            np.save(os.path.join(save_path, "tst_binary.npy"), qB)
            np.save(os.path.join(save_path, "trn_binary.npy"), rB)
            np.save(os.path.join(save_path, "trn_label.npy"), rL)
            if trn_uct is not None:
                np.save(os.path.join(save_path, "trn_uct.npy"), trn_uct)
                np.save(os.path.join(save_path, "tst_uct.npy"), tst_uct[0].detach().cpu().numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)

    if return_map == False:
        return Best_mAP
    else:
        return Best_mAP, mAP


# https://github.com/chrisbyd/DeepHash-pytorch/blob/master/validate.py
def self_validate(config, Best_mAP, val_loader, net, bit, epoch, num_dataset, return_map = False):
    device = config["device"]
    trn_binary, trn_label = compute_result(val_loader, net, config, device=device)
    mAP = self_CalcTopMap(trn_binary.numpy(), trn_label.numpy(), config["topK"])

    if mAP > Best_mAP:
        Best_mAP = mAP
        if "save_path" in config:
            save_path = os.path.join(config["save_path"], f'{config["dataset"]}_{bit}bits_{mAP}')
            os.makedirs(save_path, exist_ok=True)
            print("save in ", save_path)
            # np.save(os.path.join(save_path, "tst_label.npy"), tst_label.numpy())
            # np.save(os.path.join(save_path, "tst_binary.npy"), tst_binary.numpy())
            # np.save(os.path.join(save_path, "trn_binary.npy"), trn_binary.numpy())
            # np.save(os.path.join(save_path, "trn_label.npy"), trn_label.numpy())
            torch.save(net.state_dict(), os.path.join(save_path, "model.pt"))
    print(f"{config['info']} epoch:{epoch + 1} bit:{bit} dataset:{config['dataset']} MAP:{mAP} Best MAP: {Best_mAP}")
    print(config)

    if return_map == False:
        return Best_mAP
    else:
        return Best_mAP, mAP